import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta, binom
import os
from pathlib import Path
from datetime import datetime

# --- FISHDataAnalyzer Class ---
class FISHDataAnalyzer:
    def __init__(self, fish_data, cell_counts, selected_patterns, probe_name, calculation_method, log_callback=None):
        self.fish_data = fish_data
        self.cell_counts = cell_counts
        self.selected_patterns = selected_patterns # This is already the list of short pattern names in selected order
        self.probe_name = probe_name
        self.calculation_method = calculation_method
        self.log_callback = log_callback

    def log_message(self, message, color="black", bold=False):
        if self.log_callback:
            self.log_callback(message, color, bold)

    def reorganize_data_with_techs(self, total_cells):
        self.log_message(f"Reorganizing data for {total_cells} cells...")
        all_patterns_raw = self.fish_data['Signal_Pattern'].unique()
        pattern_mapping = {pat_raw: str(pat_raw).split('_')[-1] if '_' in str(pat_raw) else str(pat_raw) for pat_raw in all_patterns_raw}

        df = self.fish_data.copy()
        df['Signal_Pattern_Short'] = df['Signal_Pattern'].map(pattern_mapping)
        df_filtered = df[df['Tech No.'].isin(['Tech 1', 'Tech 2'])]
        return self.process_cell_count(df_filtered, total_cells)

    def process_cell_count(self, df, total_cells):
        if total_cells == self.cell_counts[0]:
            df_cells = df[df['Score No.'] == (50 if total_cells == 100 else 100)].copy()
        else: # Second count
            df_cells = df[df['Score No.'] == (100 if total_cells == 200 else [100, 150])].copy()
            if total_cells == 500: # Special handling for 500 if Score No. is [100,150]
                 df_cells = df[df['Score No.'].isin([100,150])].copy()


        # Use self.selected_patterns directly for processing relevant patterns
        # patterns_to_process are the short names the user actually selected.
        patterns_to_process = [p for p in self.selected_patterns if p in df['Signal_Pattern_Short'].unique()]
        
        self.log_message(f"Processing {len(patterns_to_process)} selected patterns for {total_cells} cells based on user selection and data availability.")

        case_info_df = df[['Case No.', 'Hybe date', 'Case ID']].drop_duplicates(subset=['Case No.'])
        result_df = pd.DataFrame(case_info_df).reset_index(drop=True)
        if 'Hybe date' in result_df.columns:
            result_df['Hybe date'] = pd.to_datetime(result_df['Hybe date'], errors='coerce').dt.strftime('%m/%d/%y')
        result_df['# Scored'] = total_cells

        for pattern in patterns_to_process: # Iterate based on selected patterns
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            result_df[tech1_col], result_df[tech2_col], result_df[pct_col] = 0, 0, 0.0

            for idx, case_row in result_df.iterrows():
                case_no_current = case_row['Case No.']
                tech1_val = df_cells[(df_cells['Case No.'] == case_no_current) & (df_cells['Signal_Pattern_Short'] == pattern) & (df_cells['Tech No.'] == 'Tech 1')]['Value'].fillna(0).sum()
                tech2_val = df_cells[(df_cells['Case No.'] == case_no_current) & (df_cells['Signal_Pattern_Short'] == pattern) & (df_cells['Tech No.'] == 'Tech 2')]['Value'].fillna(0).sum()
                sum_val = tech1_val + tech2_val
                result_df.loc[idx, tech1_col] = tech1_val
                result_df.loc[idx, tech2_col] = tech2_val
                result_df.loc[idx, pct_col] = (sum_val / total_cells) * 100 if total_cells > 0 else 0
        
        # Handle combined FRG>1swa (Break Apart) - based on selected patterns by user
        frg3_user_selected = [p for p in self.selected_patterns if 'FRG>3swa' in p or 'FRW>3swa' in p]
        frg1_3_user_selected = [p for p in self.selected_patterns if any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa'])]

        if frg3_user_selected and frg1_3_user_selected:
            self.log_message("Creating combined FRG>1swa pattern (Break Apart) as components were selected...", "blue")
            # Ensure the components' data columns were actually created if they were in patterns_to_process
            frg3_p_name = frg3_user_selected[0] # Use the first one found that user selected
            frg1_3_p_name = frg1_3_user_selected[0]

            combined_tech1_col, combined_tech2_col, combined_pct_col = "FRG>1swa_Tech 1", "FRG>1swa_Tech 2", "FRG>1swa_%"
            
            # Sum from component columns, only if those component columns exist in result_df
            val_frg3_t1 = result_df.get(f"{frg3_p_name}_Tech 1", pd.Series(0, index=result_df.index))
            val_frg1_3_t1 = result_df.get(f"{frg1_3_p_name}_Tech 1", pd.Series(0, index=result_df.index))
            result_df[combined_tech1_col] = val_frg3_t1 + val_frg1_3_t1

            val_frg3_t2 = result_df.get(f"{frg3_p_name}_Tech 2", pd.Series(0, index=result_df.index))
            val_frg1_3_t2 = result_df.get(f"{frg1_3_p_name}_Tech 2", pd.Series(0, index=result_df.index))
            result_df[combined_tech2_col] = val_frg3_t2 + val_frg1_3_t2
            
            combined_sum_values = result_df[combined_tech1_col] + result_df[combined_tech2_col]
            result_df[combined_pct_col] = (combined_sum_values / total_cells) * 100 if total_cells > 0 else 0
        return result_df

    def calculate_cutoff_and_grey_zones(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Beta method) for {total_cells} cells...")
        pattern_names_in_df = set(col.split('_')[0] for col in df.columns if '_%' in col) # Patterns present in the processed df
        
        # Filter these by what user actually selected for the report, plus combined FRG if present
        # self.selected_patterns includes short names. FRG>1swa is added if generated.
        patterns_for_cutoff = [p for p in self.selected_patterns if p in pattern_names_in_df]
        if "FRG>1swa" in pattern_names_in_df and "FRG>1swa" not in patterns_for_cutoff: # If FRG>1swa was generated
            # Check if its components were selected by user to justify including it
            frg3_user_selected = any('FRG>3swa' in sp or 'FRW>3swa' in sp for sp in self.selected_patterns)
            frg1_3_user_selected = any(any(substr in sp for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for sp in self.selected_patterns)
            if frg3_user_selected and frg1_3_user_selected:
                patterns_for_cutoff.append("FRG>1swa")
        
        results = []
        for pattern in patterns_for_cutoff: # Iterate based on selection order
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if tech1_col in df.columns and tech2_col in df.columns and pct_col in df.columns:
                sum_values = df[tech1_col] + df[tech2_col]
                max_sum = sum_values.max()
                percentages = df[pct_col].values / 100.0
                percentages = percentages[~np.isnan(percentages)]

                if len(percentages) == 0 and pattern not in df.columns: # Skip if pattern has no data at all
                    self.log_message(f"Skipping cutoff for pattern '{pattern}' as no data is available in the current sheet.", "red", True) # Changed from orange
                    continue
                
                if len(percentages) > 0 : # Ensure there is data to process
                    percentages_for_sd = percentages.copy()
                    if np.all(percentages_for_sd == 0):
                        self.log_message(f"WARNING: All percentage values are 0 for pattern {pattern}. Adding 0.5% to one case for SD calculation (Beta).", "red", True) # Changed
                        if len(percentages_for_sd) > 0: percentages_for_sd[0] = 0.005
                    
                    n_param, k_param = total_cells, min(int(max_sum if pd.notna(max_sum) else 0), total_cells -1) 
                    if k_param < 0: k_param = 0 

                    confidence_level = 0.95
                    param_a, param_b = k_param + 1, total_cells - k_param + 1 
                    if param_a <=0 : param_a = 1e-9 
                    if param_b <=0 : param_b = 1e-9

                    cutoff_beta = beta.ppf(confidence_level, param_a, param_b) if k_param <= total_cells else 1.0
                    cutoff_beta_pct = cutoff_beta * 100.0
                    
                    std_dev = np.std(percentages_for_sd) * 100.0
                    if std_dev < 0.1:
                        std_dev = 0.5
                        self.log_message(f"Applied minimum SD of 0.5% for pattern {pattern} (Beta method).", "red", True) # Changed
                    
                    grey_zone_lower = max(0, cutoff_beta_pct - 2 * std_dev)
                    grey_zone_upper = min(100, cutoff_beta_pct + 2 * std_dev)
                    # Ensure sum_values is not empty before min/max
                    valid_sum_values = sum_values.values[pd.notna(sum_values.values) & (sum_values.values >= 0)]
                    data_range = f"{int(min(valid_sum_values))}-{int(max(valid_sum_values))}" if len(valid_sum_values) > 0 else "0-0"
                    
                    results.append({
                        'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_beta_pct:.2f}%",
                        'Grey_Zone_Lower': f"{grey_zone_lower:.2f}%", 'Grey_Zone_Upper': f"{grey_zone_upper:.2f}%",
                        'Range': data_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"
                    })
                else: # If percentages array is empty after filtering NaNs
                    self.log_message(f"No valid percentage data to calculate Beta cutoff for pattern '{pattern}'.", "red", True) # Changed

        self.log_message(f"Calculated {len(results)} Beta cutoff values for {total_cells} cells.")
        return pd.DataFrame(results)

    def calculate_cutoff_and_grey_zones_cribinom(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Cribinom method) for {total_cells} cells...")
        pattern_names_in_df = set(col.split('_')[0] for col in df.columns if '_%' in col)
        
        patterns_for_cutoff = [p for p in self.selected_patterns if p in pattern_names_in_df]
        if "FRG>1swa" in pattern_names_in_df and "FRG>1swa" not in patterns_for_cutoff:
            frg3_user_selected = any('FRG>3swa' in sp or 'FRW>3swa' in sp for sp in self.selected_patterns)
            frg1_3_user_selected = any(any(substr in sp for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for sp in self.selected_patterns)
            if frg3_user_selected and frg1_3_user_selected:
                patterns_for_cutoff.append("FRG>1swa")

        results = []
        confidence_level = 0.95
        for pattern in patterns_for_cutoff:
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if tech1_col in df.columns and tech2_col in df.columns and pct_col in df.columns:
                sum_values = df[tech1_col] + df[tech2_col]
                percentages = df[pct_col].values / 100.0
                percentages = percentages[~np.isnan(percentages)]

                if len(percentages) == 0 and pattern not in df.columns:
                    self.log_message(f"Skipping cutoff for pattern '{pattern}' as no data is available in the current sheet.", "red", True) # Changed
                    continue

                if len(percentages) > 0:
                    p_for_cutoff = np.mean(percentages)
                    if np.all(percentages == 0) or p_for_cutoff == 0:
                        p_for_cutoff = 0.005 
                        self.log_message(f"WARNING: All percentages are 0 or mean is 0 for pattern '{pattern}'. Using p=0.005 for Cribinom.", "red", True) # Changed

                    cutoff_count = binom.ppf(confidence_level, n=total_cells, p=p_for_cutoff)
                    if cutoff_count == 0 and p_for_cutoff < 0.01: 
                         cutoff_count = 1 
                         self.log_message(f"Adjusted Cribinom cutoff for '{pattern}' from 0 to 1.", "red", True) # Changed

                    cutoff_cribinom_pct = (cutoff_count / total_cells) * 100.0
                    grey_zone_lower = cutoff_cribinom_pct 
                    grey_zone_upper = min(100, cutoff_cribinom_pct + 3.0)
                    
                    std_dev = np.std(percentages) * 100.0
                    if std_dev < 0.1 and np.any(percentages > 0): # only apply if some data exists
                        std_dev = 0.5
                        self.log_message(f"Applied minimum SD of 0.5% for pattern {pattern} (Cribinom method).", "red", True) # Changed
                    elif np.all(percentages == 0): std_dev = 0.0 # if all percentages are zero, std dev is zero
                        
                    valid_sum_values = sum_values.values[pd.notna(sum_values.values) & (sum_values.values >= 0)]
                    data_range = f"{int(min(valid_sum_values))}-{int(max(valid_sum_values))}" if len(valid_sum_values) > 0 else "0-0"
                    results.append({
                        'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_cribinom_pct:.2f}%",
                        'Grey_Zone_Lower': f"{grey_zone_lower:.2f}%", 'Grey_Zone_Upper': f"{grey_zone_upper:.2f}%",
                        'Range': data_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"
                    })
                else:
                    self.log_message(f"No valid percentage data to calculate Cribinom cutoff for pattern '{pattern}'.", "red", True) # Changed
        self.log_message(f"Calculated {len(results)} Cribinom cutoff values for {total_cells} cells.")
        return pd.DataFrame(results)

    def create_reorganized_html_content(self, df_first, df_second):
        # ... (HTML structure remains largely the same) ...
        table_width, font_size = "100%", "10px"
        html_content = f"""<!DOCTYPE html><html><head>
            <title>{self.probe_name} Cell Analysis ({self.cell_counts[0]}/{self.cell_counts[1]} Cells)</title><style>
            body{{font-family:Arial,sans-serif;margin:0;padding:20px;background-color:#f5f5f5;font-size:{font_size};}}
            .main-container{{max-width:{table_width};margin:0 auto;overflow-x:auto;}}
            .table-section{{margin-bottom:50px;background-color:white;padding:20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);overflow-x:auto;}}
            h1{{text-align:center;color:#333;margin-bottom:30px;font-size:24px;}}
            h2{{text-align:center;color:#007bff;margin-bottom:20px;font-size:18px;border-bottom:2px solid #007bff;padding-bottom:10px;}}
            table{{width:100%;border-collapse:collapse;margin:20px 0;background-color:white;table-layout:auto;min-width:1000px;}}
            th,td{{border:1px solid #ddd;padding:5px;text-align:center;vertical-align:middle;white-space:nowrap;}}
            .header-light-blue{{background-color:#3498db;color:white;font-weight:bold;}}
            .header-dark-blue{{background-color:#34495e;color:white;font-weight:bold;}}
            .highlight{{background-color:#dc3545 !important;color:white !important;font-weight:bold;}}
            .data-row:nth-child(even){{background-color:#f8f9fa;}}.data-row:hover{{background-color:#e3f2fd;}}
            @media print{{body{{margin:0;padding:0;}}.main-container{{max-width:none;width:100%;overflow-x:visible;}}
            .table-section{{box-shadow:none;padding:0;margin-bottom:20px;}}table{{min-width:auto;width:100%;}}}}
            @media screen and (max-width:768px){{.main-container{{max-width:100%;padding:10px;}}
            .table-section{{padding:10px;}}table{{font-size:8px;min-width:auto;}}th,td{{padding:2px;}}}}</style></head><body>
            <div class="main-container"><h1>{self.probe_name} Cell Analysis Report ({self.cell_counts[0]} and {self.cell_counts[1]} Cells)</h1>"""
        if df_first is not None and not df_first.empty:
            html_content += f'<div class="table-section"><h2>{self.probe_name} {self.cell_counts[0]} Cells Data</h2>{self.create_table_content(df_first, self.cell_counts[0])}</div>'
        if df_second is not None and not df_second.empty:
            html_content += f'<div class="table-section"><h2>{self.probe_name} {self.cell_counts[1]} Cells Data</h2>{self.create_table_content(df_second, self.cell_counts[1])}</div>'
        html_content += "</div></body></html>"
        return html_content

    def create_table_content(self, df, total_cells): # df is the reorganized data for one cell count
        if df.empty: return "<p>No data available for table content generation.</p>"
        base_cols = ['Case No.', 'Hybe date', 'Case ID', '# Scored']
        
        # Determine all unique short pattern names available in the current df's columns
        all_patterns_in_data = list(set(col.split('_')[0] for col in df.columns if col.endswith('_Tech 1') or col.endswith('_Tech 2') or col.endswith('_')))

        # Use self.selected_patterns (which is in user's selection order)
        # and filter by patterns actually present in the current data (df)
        final_pattern_names_for_table = [
            p for p in self.selected_patterns if p in all_patterns_in_data
        ]
        
        # Handle combined FRG>1swa: if its components were selected by user AND it was generated (exists in all_patterns_in_data)
        # and it's not already part of final_pattern_names_for_table (e.g. if user explicitly selected a pattern named "FRG>1swa")
        frg3_user_selected = any('FRG>3swa' in p or 'FRW>3swa' in p for p in self.selected_patterns)
        frg1_3_user_selected = any(any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for p in self.selected_patterns)

        if frg3_user_selected and frg1_3_user_selected and "FRG>1swa" in all_patterns_in_data:
            if "FRG>1swa" not in final_pattern_names_for_table:
                final_pattern_names_for_table.append("FRG>1swa") # Append to maintain prior order

        if not final_pattern_names_for_table:
             return "<p>No selected patterns have corresponding data to display in table.</p>"

        max_values, max_row_indices = {}, {}
        for pattern in final_pattern_names_for_table:
            pct_col = f"{pattern}_%"
            if pct_col in df.columns: # Ensure the percentage column exists
                max_val = df[pct_col].max()
                if pd.notna(max_val) and max_val > 0:
                    max_values[pattern] = max_val
                    # Get the first index if multiple rows have the max value
                    max_indices = df[df[pct_col] == max_val].index
                    if not max_indices.empty:
                        max_row_indices[pattern] = max_indices[0]
        
        html = "<table><thead><tr>"
        for col in base_cols: html += f'<th rowspan="2" class="header-light-blue">{col}</th>'
        for pattern in final_pattern_names_for_table: html += f'<th colspan="3" class="header-dark-blue">{pattern}</th>'
        html += '</tr><tr>'
        for _ in final_pattern_names_for_table: html += '<th class="header-dark-blue">Tech 1</th><th class="header-dark-blue">Tech 2</th><th class="header-dark-blue">%</th>'
        html += '</tr></thead><tbody>'
        for row_idx, row_data in df.iterrows():
            html += '<tr class="data-row">'
            for col in base_cols: html += f'<td>{row_data.get(col, "")}</td>'
            for pattern in final_pattern_names_for_table:
                t1, t2, pct = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
                v1, v2, vp = row_data.get(t1, 0), row_data.get(t2, 0), row_data.get(pct, 0.0)
                highlight_class = ""
                if pattern in max_values and max_row_indices.get(pattern) == row_idx and vp > 0: # Check if current row_idx matches stored max_row_idx
                    if pd.notna(vp) and vp == max_values.get(pattern): # Ensure value matches
                         highlight_class = "highlight"
                html += f'<td class="{highlight_class}">{v1}</td><td class="{highlight_class}">{v2}</td><td class="{highlight_class}">{vp:.2f}%</td>'
            html += '</tr>'
        html += "</tbody></table>"
        return html

    def create_cutoff_html_content(self, cutoff_first, cutoff_second):
        # ... (HTML structure remains largely the same) ...
        # The cutoff DataFrames (cutoff_first, cutoff_second) should already be in the desired order
        # if calculate_cutoff_and_grey_zones functions iterate based on self.selected_patterns.
        html_content = f"""<!DOCTYPE html><html><head>
            <title>{self.probe_name} Cut-off/GrayZone ({self.cell_counts[0]}/{self.cell_counts[1]} Cells) - {self.calculation_method}</title><style>
            body{{font-family:Arial,sans-serif;margin:10px;background-color:#f5f5f5;font-size:14px;}}
            .main-container{{max-width:900px;margin:0 auto;background-color:white;padding:15px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}}
            .title-container{{display:flex;justify-content:center;margin-bottom:15px;}}
            h1{{color:#333;margin:0;font-size:18px;border-bottom:1px solid #007bff;padding-bottom:8px;display:inline-block;}}
            .tables-container{{display:flex;gap:20px;justify-content:space-around;align-items:flex-start;flex-wrap:wrap;}}
            .table-section{{flex:1;min-width:300px;margin-bottom:20px;}}
            .section-title{{text-align:center;color:#007bff;margin-bottom:15px;font-size:16px;background-color:#e3f2fd;padding:8px;border-radius:5px;font-weight:bold;}}
            .patterns-container{{display:flex;flex-direction:column;gap:12px;}}
            .pattern-container{{border:1px solid #ddd;border-radius:5px;overflow:hidden;background-color:white;}}
            .pattern-header{{background-color:#f8f9fa;padding:6px 10px;font-weight:bold;font-size:13px;color:#333;border-bottom:1px solid #ddd;text-align:center;}}
            .compact-table{{width:100%;border-collapse:collapse;font-size:12px;}}
            .compact-table td{{padding:5px 8px;border-bottom:1px solid #eee;text-align:left;}}
            .compact-table tr:last-child td{{border-bottom:none;}}
            .label-cell{{background-color:#f8f9fa;font-weight:bold;width:45%;color:#555;}}
            .value-cell{{background-color:white;width:55%;}}
            .cutoff-value{{color:#dc3545;font-weight:bold;}}.grey-zone{{color:#6c757d;}}
            .range-value{{color:#28a745;font-weight:bold;}}.std-value{{color:#17a2b8;}}
            .controls{{color:#6f42c1;font-style:italic;font-size:10px;}}
            .no-data{{text-align:center;color:#6c757d;font-style:italic;padding:20px;background-color:#f8f9fa;border-radius:5px;}}
            @media screen and (max-width:768px){{.tables-container{{flex-direction:column;align-items:center;}}}}</style></head><body>
            <div class="main-container"><div class="title-container"><h1>{self.probe_name} Cut-off/GrayZone Report ({self.cell_counts[0]}&{self.cell_counts[1]} Cells) - {self.calculation_method} Method</h1></div><div class="tables-container">"""
        html_content += '<div class="table-section">'
        html_content += f'<div class="section-title">{self.probe_name} {self.cell_counts[0]} Database</div><div class="patterns-container">'
        if cutoff_first is not None and not cutoff_first.empty: html_content += self.create_compact_pattern_sections(cutoff_first)
        else: html_content += f'<div class="no-data">No data for {self.cell_counts[0]} cells</div>'
        html_content += '</div></div>'
        html_content += '<div class="table-section">'
        html_content += f'<div class="section-title">{self.probe_name} {self.cell_counts[1]} Database</div><div class="patterns-container">'
        if cutoff_second is not None and not cutoff_second.empty: html_content += self.create_compact_pattern_sections(cutoff_second)
        else: html_content += f'<div class="no-data">No data for {self.cell_counts[1]} cells</div>'
        html_content += '</div></div></div></div></body></html>'
        return html_content

    def create_compact_pattern_sections(self, df_results): # df_results is one of the cutoff DFs
        html = ""
        # The df_results should already be in the correct order from calculate_cutoff functions
        for _, row in df_results.iterrows(): # Iterating over DataFrame rows preserves their order
            html += f"""<div class="pattern-container"><div class="pattern-header">{row['Signal_Pattern']}</div>
                <table class="compact-table">
                <tr><td class="label-cell">Cutoff (95% C.I.)</td><td class="value-cell cutoff-value">{row['Cutoff_95%_CI']}</td></tr>
                <tr><td class="label-cell">Gray Zone</td><td class="value-cell grey-zone">{row['Grey_Zone_Lower']} to {row['Grey_Zone_Upper']}</td></tr>
                <tr><td class="label-cell">Range</td><td class="value-cell range-value">{row['Range']}</td></tr>
                <tr><td class="label-cell">Std. Dev.</td><td class="value-cell std-value">{row['Standard_Deviation']} <span class="controls">{row['Controls']}</span></td></tr>
                </table></div>"""
        return html

# === Helper Functions ===
def streamlit_log_message(message, color="black", bold=False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry_text = f"[{timestamp}] {message}"
    
    styled_log_entry = log_entry_text
    
    # Apply bold first if requested
    if bold:
        styled_log_entry = f"**{log_entry_text}**"
    
    # Then apply color markdown. If bold was applied, it wraps the bolded text.
    # For "orange" warnings, user wants red and bold.
    if color == "orange": # Treat "orange" as a directive for red & bold warning
        styled_log_entry = f":red[**{log_entry_text}**]" # Ensure bold, make it red
    elif color == "red":
        styled_log_entry = f":red[{styled_log_entry}]" # Already bolded if bold=True
    elif color == "green":
        styled_log_entry = f":green[{styled_log_entry}]"
    elif color == "blue":
        styled_log_entry = f":blue[{styled_log_entry}]"
    # If color is "black" or unhandled, styled_log_entry remains as is (plain or bolded)
    
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(styled_log_entry)


def process_single_table_excel(df_raw, start, end, log_func):
    try:
        header_start = max(0, start - 1) 
        table_with_headers = df_raw.iloc[header_start:end+1].copy().reset_index(drop=True)

        if table_with_headers.shape[0] < 2: 
            log_func(f"Table at original rows {start+1}-{end+1} is too short for header processing.", "red", True) # Changed
            return []

        merged_header_row = table_with_headers.iloc[0].tolist()
        specific_header_row = table_with_headers.iloc[1].tolist() 

        column_categories = {}
        current_category = ""
        for i, header_val in enumerate(merged_header_row):
            if pd.notna(header_val) and str(header_val).strip():
                current_category = str(header_val).strip()
            column_categories[i] = current_category
        
        combined_headers = []
        for i, specific_header in enumerate(specific_header_row):
            specific_header_str = str(specific_header).strip()
            parent_category = column_categories.get(i, "")
            if specific_header_str:
                if parent_category and ("Signal patterns" in parent_category or "Other" in parent_category):
                    combined_headers.append(f"{parent_category}_{specific_header_str}")
                else:
                    combined_headers.append(specific_header_str)
            else: 
                combined_headers.append(f"UnnamedCol_{i}")

        table_data_rows = df_raw.iloc[start+1:end+1].copy() 
        table_header_row_df = df_raw.iloc[start:start+1].copy()

        current_headers = table_header_row_df.iloc[0].tolist()
        final_headers = []

        for i, current_header_val in enumerate(current_headers):
            ch_str = str(current_header_val).strip()
            if i < len(combined_headers) and ("Signal patterns" in combined_headers[i] or "Other" in combined_headers[i]):
                final_headers.append(combined_headers[i])
            elif ch_str : 
                final_headers.append(ch_str)
            else: 
                final_headers.append(f"UnnamedCol_{i}")
        
        if len(final_headers) != table_data_rows.shape[1]:
            log_func(f"Header length mismatch in table at original rows {start+1}-{end+1}. Expected {table_data_rows.shape[1]}, got {len(final_headers)}. Adjusting.", "red", True) # Changed
            final_headers = final_headers[:table_data_rows.shape[1]] 
            while len(final_headers) < table_data_rows.shape[1]: 
                final_headers.append(f"AutoHeader_{len(final_headers)}")


        table_data_rows.columns = final_headers
        table_data_rows = table_data_rows.reset_index(drop=True)

        tech_cols_fill = [col for col in ["Tech No.", "Tech Initial"] if col in table_data_rows.columns]
        if tech_cols_fill:
            table_data_rows[tech_cols_fill] = table_data_rows[tech_cols_fill].ffill()

        case_no = table_data_rows["Case No."].iloc[0] if "Case No." in final_headers and not table_data_rows["Case No."].empty and pd.notna(table_data_rows["Case No."].iloc[0]) else None
        hybe_date = table_data_rows["Hybe date"].iloc[0] if "Hybe date" in final_headers and not table_data_rows["Hybe date"].empty and pd.notna(table_data_rows["Hybe date"].iloc[0]) else None
        case_id = table_data_rows["Case ID"].iloc[0] if "Case ID" in final_headers and not table_data_rows["Case ID"].empty and pd.notna(table_data_rows["Case ID"].iloc[0]) else None

        signal_pattern_cols = [col for col in final_headers if isinstance(col, str) and ("Signal patterns" in col or "Other" in col)]
        processed_data_list = []

        if "Tech No." in final_headers:
            for tech_num, group in table_data_rows.groupby("Tech No."):
                tech_initial = group["Tech Initial"].iloc[0] if "Tech Initial" in final_headers and pd.notna(group["Tech Initial"].iloc[0]) else None
                for _, score_row in group.iterrows():
                    if "Score No." in final_headers and pd.notna(score_row["Score No."]):
                        score_no_val = score_row["Score No."]
                        score_sum_val = score_row["Score Sum"] if "Score Sum" in final_headers and pd.notna(score_row["Score Sum"]) else None
                        for sig_col_name in signal_pattern_cols:
                            if sig_col_name in score_row: 
                                value = score_row[sig_col_name] if pd.notna(score_row[sig_col_name]) else 0
                                data_point = {
                                    "Case No.": case_no, "Hybe date": hybe_date, "Case ID": case_id,
                                    "Tech No.": tech_num, "Tech Initial": tech_initial,
                                    "Score No.": score_no_val, "Score Sum": score_sum_val,
                                    "Signal_Pattern": sig_col_name, "Value": value
                                }
                                processed_data_list.append(pd.DataFrame([data_point]))
        else:
            log_func(f"Warning: 'Tech No.' column not found in table at original rows {start+1}-{end+1}. Cannot process tech-specific data.", "red", True) # Changed
        return processed_data_list
    except Exception as e:
        log_func(f"CRITICAL error processing table at original rows {start+1}-{end+1}: {e}", "red", True)
        import traceback
        log_func(traceback.format_exc(),"red") 
        return []


def is_valid_table_excel(df_raw, start, end, log_func):
    try:
        table_segment = df_raw.iloc[start:end+1]
        data_rows = table_segment.iloc[1:] 
        if data_rows.empty: return False
        
        valid_value_count, total_cells = 0, 0
        for col_idx in data_rows.columns:
            for value in data_rows[col_idx]:
                total_cells += 1
                if pd.notna(value) and str(value).strip() != "":
                    if isinstance(value, (int, float)) and value != 0:
                        valid_value_count += 1
                    elif not isinstance(value, (int, float)): 
                         valid_value_count += 1
        
        if total_cells == 0: return False
        return (valid_value_count / total_cells) >= 0.05 
    except Exception as e:
        log_func(f"Error during table validation (rows {start+1}-{end+1}): {e}", "red")
        return True

def read_fish_data_from_excel(uploaded_file_obj, log_func, progress_bar_st):
    try:
        log_func("Reading Excel sheet...")
        df_raw = pd.read_excel(uploaded_file_obj, sheet_name=0, header=None)
        
        table_starts = df_raw[df_raw.iloc[:, 0] == "Case No."].index.tolist()
        if not table_starts:
            log_func("No tables found with 'Case No.' in the first column. Ensure Excel format is correct.", "red", True)
            return pd.DataFrame()

        table_ends = table_starts[1:] + [len(df_raw)] 
        table_ends = [end - 1 for end in table_ends] 

        log_func(f"Found {len(table_starts)} potential tables based on 'Case No.'.")
        all_tables_data_list = []
        excluded_count = 0
        
        for i, (start_idx, end_idx) in enumerate(zip(table_starts, table_ends)):
            if progress_bar_st:
                progress_val = (i + 1) / len(table_starts)
                progress_bar_st.progress(max(0.0, min(1.0, progress_val)), text=f"Processing Excel table {i+1}/{len(table_starts)}...")

            if start_idx >= len(df_raw) or end_idx >= len(df_raw) or start_idx > end_idx:
                log_func(f"Skipping invalid table indices: start={start_idx}, end={end_idx}", "red")
                excluded_count += 1
                continue

            if is_valid_table_excel(df_raw, start_idx, end_idx, log_func):
                processed_parts = process_single_table_excel(df_raw, start_idx, end_idx, log_func)
                if processed_parts:
                    all_tables_data_list.extend(processed_parts)
                    log_func(f"Table {i+1} (original rows {start_idx+1}-{end_idx+1}) processed.")
            else:
                excluded_count += 1
                log_func(f"Table {i+1} (original rows {start_idx+1}-{end_idx+1}) excluded due to insufficient valid data.", "red", True) # Changed
        
        if excluded_count > 0:
            log_func(f"INFO: Excluded {excluded_count} tables.", "blue")
        
        if all_tables_data_list:
            result_df = pd.concat(all_tables_data_list, ignore_index=True)
            if "Hybe date" in result_df.columns:
                result_df["Hybe date"] = pd.to_datetime(result_df["Hybe date"], errors='coerce')
            log_func(f"Successfully combined data from processed tables. Total records: {len(result_df)}.", "green", True)
            return result_df
        else:
            log_func("No valid data tables could be extracted from the Excel file.", "red", True)
            return pd.DataFrame()
    except Exception as e:
        log_func(f"CRITICAL error reading/processing Excel: {e}", "red", True)
        import traceback
        log_func(traceback.format_exc(),"red")
        raise 

def detect_cell_counts_streamlit(fish_data_df, log_func):
    if fish_data_df is None or fish_data_df.empty: return [200, 500] 
    score_nos = fish_data_df['Score No.'].dropna().unique()
    log_func(f"Found Score No. values: {sorted(list(score_nos))}")
    
    counts = [200, 500] 
    if 50.0 in score_nos or 50 in score_nos: # Check for float and int
        counts = [100, 200] 
    elif (100.0 in score_nos or 100 in score_nos) and \
         (150.0 in score_nos or 150 in score_nos): 
        counts = [200, 500] 
    elif 100.0 in score_nos or 100 in score_nos: 
        counts = [200, 500]
    log_func(f"Determined cell counts: {counts}")
    return counts

def get_available_patterns_streamlit(fish_data_df, log_func):
    if fish_data_df is None or fish_data_df.empty: return []
    unique_patterns_raw = fish_data_df['Signal_Pattern'].unique()
    pattern_map = {p_raw: str(p_raw).split('_')[-1] if '_' in str(p_raw) else str(p_raw) for p_raw in unique_patterns_raw}
    available_short_patterns = sorted(list(set(pattern_map.values())))
    log_func(f"Available short signal patterns: {available_short_patterns}")
    return available_short_patterns

def check_frg_pattern_combination_streamlit(selected_patterns_short, log_func):
    frg3 = any('FRG>3swa' in p or 'FRW>3swa' in p for p in selected_patterns_short)
    frg1_3 = any(any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for p in selected_patterns_short)
    if frg3 and frg1_3:
        msg = "Break_apart probe patterns selected. Software will automatically generate combined 'FRG>1swa' results."
        st.info(msg) # This is a streamlit info box, not a log message directly.
        log_func("FRG (Break Apart) pattern combination detected - will generate combined results.", "blue")
        return True
    return False

# --- Streamlit App ---
def run_fish_analysis_app():
    st.set_page_config(page_title="FISH Analysis Tool", layout="wide")
    st.title("🔬 FISH Analysis Tool 📊")

    # Initialize session state
    for key, default_val in [
        ('log_messages', []), ('fish_data', None), ('available_patterns', []),
        ('cell_counts', [200, 500]), ('probe_name', ""),
        ('selected_patterns_ui', []), # Canonical list of selected patterns
        ('data_loaded_successfully', False), ('last_uploaded_filename', None),
        ('current_probe_name_in_widget', ""),
        ('html_reorganized_content', None), ('fname_reorganized_report', None),
        ('html_cutoff_content', None), ('fname_cutoff_report', None),
        ('reports_generated_once', False)
    ]:
        if key not in st.session_state: st.session_state[key] = default_val

    # --- Callback for pattern selection changes ---
    def multiselect_on_change_callback():
        # Update the canonical selected_patterns_ui from the widget's state
        if 'pattern_multiselect_widget' in st.session_state: # key of the multiselect
            st.session_state.selected_patterns_ui = st.session_state.pattern_multiselect_widget
        
        # Common logic for any selection change
        st.session_state.reports_generated_once = False
        streamlit_log_message("Pattern selection changed. Reports will need to be regenerated.", "blue")

    # --- Log Display Area (Sidebar) ---
    with st.sidebar:
        st.header("📋 Processing Log")
        log_placeholder = st.empty()
        if st.button("Clear Log", key="clear_log_sidebar"):
            st.session_state.log_messages = []
            streamlit_log_message("Log cleared by user.")

    # --- Step 1: File Upload and Setup ---
    st.header("Step 1: File Upload and Probe Setup")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx, .xls)", type=["xlsx", "xls"], key="file_uploader_widget")

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.data_loaded_successfully = False 
            st.session_state.fish_data = None
            st.session_state.available_patterns = []
            st.session_state.selected_patterns_ui = [] # Reset canonical list
            if 'pattern_multiselect_widget' in st.session_state: # Reset widget state if it exists
                st.session_state.pattern_multiselect_widget = []
            
            st.session_state.html_reorganized_content = None
            st.session_state.fname_reorganized_report = None
            st.session_state.html_cutoff_content = None
            st.session_state.fname_cutoff_report = None
            st.session_state.reports_generated_once = False

            filename_stem = Path(uploaded_file.name).stem
            potential_probe = filename_stem.split('_')[0] if '_' in filename_stem else \
                              filename_stem.split('-')[0] if '-' in filename_stem else filename_stem
            st.session_state.current_probe_name_in_widget = potential_probe.upper()
            streamlit_log_message(f"Inferred probe name: {st.session_state.current_probe_name_in_widget}", "blue")
    
    st.session_state.probe_name = st.text_input(
        "Probe Name (e.g., HER2, ALK)",
        value=st.session_state.current_probe_name_in_widget,
        key="probe_name_text_input",
        on_change=lambda: setattr(st.session_state, 'current_probe_name_in_widget', st.session_state.probe_name_text_input.upper().strip())
    ).upper().strip()

    if st.button("Load and Process Data", key="load_data_main_button"):
        st.session_state.log_messages = [] 
        st.session_state.html_reorganized_content = None
        st.session_state.fname_reorganized_report = None
        st.session_state.html_cutoff_content = None
        st.session_state.fname_cutoff_report = None
        st.session_state.reports_generated_once = False
        st.session_state.selected_patterns_ui = [] # Reset selections on new load
        if 'pattern_multiselect_widget' in st.session_state:
             st.session_state.pattern_multiselect_widget = []


        streamlit_log_message("Attempting to load and process data...")
        if not uploaded_file:
            st.error("❌ Please upload an Excel file first.")
            streamlit_log_message("Error: No Excel file uploaded.", "red", True)
        elif not st.session_state.probe_name:
            st.error("❌ Probe name is required. Please enter a probe name.")
            streamlit_log_message("Error: Probe name missing.", "red", True)
        else:
            streamlit_log_message(f"Processing for probe: {st.session_state.probe_name}", "blue")
            load_progress_bar = st.progress(0.0, text="Initializing data load...")
            try:
                with st.spinner("Reading and analyzing Excel file... This might take a moment."):
                    st.session_state.fish_data = read_fish_data_from_excel(uploaded_file, streamlit_log_message, load_progress_bar)
                
                if st.session_state.fish_data is not None and not st.session_state.fish_data.empty:
                    load_progress_bar.progress(0.75, text="Detecting cell counts and patterns...")
                    st.session_state.cell_counts = detect_cell_counts_streamlit(st.session_state.fish_data, streamlit_log_message)
                    st.session_state.available_patterns = get_available_patterns_streamlit(st.session_state.fish_data, streamlit_log_message)
                    load_progress_bar.progress(1.0, text="Data loaded successfully!")
                    st.success(f"✅ Data loaded! {len(st.session_state.fish_data)} records. Counts: {st.session_state.cell_counts}. Patterns: {len(st.session_state.available_patterns)}.")
                    st.session_state.data_loaded_successfully = True
                    if not st.session_state.available_patterns:
                        st.warning("⚠️ No signal patterns were identified in the data. Check file content and format.") # This is st.warning, not streamlit_log_message
                else:
                    st.error("❌ No valid data extracted. Check logs for details.")
                    st.session_state.data_loaded_successfully = False
            except Exception as e:
                st.error(f"💥 Critical error during data loading: {e}")
                streamlit_log_message(f"CRITICAL LOAD ERROR: {e}", "red", True)
                st.session_state.data_loaded_successfully = False
            finally:
                if 'load_progress_bar' in locals() : load_progress_bar.empty()

    # --- Step 2: Signal Pattern Selection ---
    st.header("Step 2: Signal Pattern Selection")
    if st.session_state.data_loaded_successfully and st.session_state.available_patterns:
        col1_btn, col2_btn = st.columns(2)
        if col1_btn.button("Select All Patterns", key="select_all_btn"):
            st.session_state.selected_patterns_ui = st.session_state.available_patterns[:]
            st.session_state.pattern_multiselect_widget = st.session_state.selected_patterns_ui # Update widget state for next render
            st.session_state.reports_generated_once = False
            streamlit_log_message("Selected all available patterns. Reports will need to be regenerated.", "blue")
            st.rerun() # Rerun to make multiselect reflect the change immediately

        if col2_btn.button("Clear All Selections", key="clear_all_btn"):
            st.session_state.selected_patterns_ui = []
            st.session_state.pattern_multiselect_widget = [] # Update widget state
            st.session_state.reports_generated_once = False
            streamlit_log_message("Cleared all pattern selections. Reports will need to be regenerated.", "blue")
            st.rerun() # Rerun for immediate reflection

        st.multiselect(
            "Select Signal Patterns:",
            options=st.session_state.available_patterns,
            default=st.session_state.selected_patterns_ui, # Default to our canonical list
            key="pattern_multiselect_widget", # This key's value in session_state is the widget's current selection
            on_change=multiselect_on_change_callback
        )
    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot select patterns.")
    else:
        st.info("Load data in Step 1 to enable pattern selection.")

    # --- Step 3: Generate Reports ---
    st.header("Step 3: Generate Reports")
    if st.session_state.data_loaded_successfully and st.session_state.available_patterns:
        if not st.session_state.selected_patterns_ui: # Check the canonical list
            st.warning("⚠️ Please select at least one signal pattern in Step 2 to generate reports.")
        else:
            method_label = st.radio(
                "Select Cut-off Calculation Method:",
                ("Beta Inverse Function (Original Method)", "CRITBINOM Function (Alternative Method)"),
                format_func=lambda x: x.split(" (")[0],
                key="calc_method_radio_main"
            )
            calc_method_actual = "Beta" if "Beta" in method_label else "Cribinom"

            if st.button("Generate HTML Reports", key="generate_reports_main_button"):
                # Use st.session_state.selected_patterns_ui as it's the canonical list
                streamlit_log_message(f"Generating reports for: {st.session_state.selected_patterns_ui}, Probe: {st.session_state.probe_name}, Method: {calc_method_actual}", "blue", True)
                check_frg_pattern_combination_streamlit(st.session_state.selected_patterns_ui, streamlit_log_message)
                
                report_progress_bar = st.progress(0.0, text="Initializing report generation...")
                try:
                    with st.spinner("Generating reports... This may take a few moments."):
                        analyzer = FISHDataAnalyzer(
                            st.session_state.fish_data, st.session_state.cell_counts, 
                            st.session_state.selected_patterns_ui, # Pass the canonical list
                            st.session_state.probe_name,
                            calc_method_actual, streamlit_log_message
                        )
                        
                        report_progress_bar.progress(0.1, text="Generating reorganized data...")
                        df_first_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[0])
                        df_second_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[1])
                        html_reorganized = analyzer.create_reorganized_html_content(df_first_reorg, df_second_reorg)
                        fname_reorg = f"{analyzer.probe_name}_reorganized_data_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"
                        
                        st.session_state.html_reorganized_content = html_reorganized
                        st.session_state.fname_reorganized_report = fname_reorg
                        
                        report_progress_bar.progress(0.5, text="Generating cutoff values data...")
                        if calc_method_actual == "Beta":
                            cut_first = analyzer.calculate_cutoff_and_grey_zones(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones(df_second_reorg, analyzer.cell_counts[1])
                        else: # Cribinom
                            cut_first = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_second_reorg, analyzer.cell_counts[1])
                        html_cutoff = analyzer.create_cutoff_html_content(cut_first, cut_second)
                        fname_cutoff = f"{analyzer.probe_name}_{calc_method_actual.lower()}_cutoff_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"

                        st.session_state.html_cutoff_content = html_cutoff
                        st.session_state.fname_cutoff_report = fname_cutoff
                        
                        st.session_state.reports_generated_once = True
                        report_progress_bar.progress(1.0, text="Reports generated!")
                        st.success("✅ Reports generated successfully!")
                        
                except Exception as e:
                    st.error(f"💥 Failed to generate reports: {e}")
                    streamlit_log_message(f"CRITICAL REPORTING ERROR: {e}", "red", True)
                    import traceback
                    streamlit_log_message(traceback.format_exc(),"red")
                    st.session_state.reports_generated_once = False
                finally:
                   if 'report_progress_bar' in locals(): report_progress_bar.empty()

            if st.session_state.get('reports_generated_once', False):
                dl_col1, dl_col2 = st.columns(2)
                if st.session_state.get('html_reorganized_content') and st.session_state.get('fname_reorganized_report'):
                    with dl_col1:
                        st.download_button(
                            label=f"📥 Download Reorganized Data ({st.session_state.fname_reorganized_report})",
                            data=st.session_state.html_reorganized_content,
                            file_name=st.session_state.fname_reorganized_report,
                            mime="text/html",
                            key="dl_reorg_persistent" 
                        )
                if st.session_state.get('html_cutoff_content') and st.session_state.get('fname_cutoff_report'):
                    with dl_col2:
                        st.download_button(
                            label=f"📥 Download Cutoff Values ({st.session_state.fname_cutoff_report})",
                            data=st.session_state.html_cutoff_content,
                            file_name=st.session_state.fname_cutoff_report,
                            mime="text/html",
                            key="dl_cutoff_persistent"
                        )
    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot generate reports.")
    else:
        st.info("Load data and select patterns in Steps 1 & 2 to enable report generation.")

    # Update log display
    log_output = "\n\n".join(st.session_state.log_messages)
    log_placeholder.markdown(log_output if log_output else "No log messages yet.", unsafe_allow_html=False)

if __name__ == "__main__":
    run_fish_analysis_app()
