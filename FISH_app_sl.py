import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta, binom
import os
from pathlib import Path
from datetime import datetime

# --- FISHDataAnalyzer Class (Copied directly from your provided code) ---
class FISHDataAnalyzer:
    def __init__(self, fish_data, cell_counts, selected_patterns, probe_name, calculation_method, log_callback=None):
        """Initialize the analyzer with data and parameters"""
        self.fish_data = fish_data
        self.cell_counts = cell_counts  # e.g., [100, 200] or [200, 500]
        self.selected_patterns = selected_patterns
        self.probe_name = probe_name
        self.calculation_method = calculation_method # "Beta" or "Cribinom"
        self.log_callback = log_callback

    def log_message(self, message, color="black", bold=False):
        """Log message if callback is available"""
        if self.log_callback:
            self.log_callback(message, color, bold)

    def reorganize_data_with_techs(self, total_cells):
        """Reorganize data for specific cell count"""
        self.log_message(f"Reorganizing data for {total_cells} cells...")

        all_patterns = self.fish_data['Signal_Pattern'].unique()
        pattern_mapping = {}

        for pattern in all_patterns:
            if '_' in str(pattern):
                pattern_mapping[pattern] = pattern.split('_')[-1]
            else:
                pattern_mapping[pattern] = pattern

        df = self.fish_data.copy()
        df['Signal_Pattern_Short'] = df['Signal_Pattern'].map(pattern_mapping)
        df_filtered = df[df['Tech No.'].isin(['Tech 1', 'Tech 2'])]
        return self.process_cell_count(df_filtered, total_cells)

    def process_cell_count(self, df, total_cells):
        """Process data for a specific cell count"""
        if total_cells == self.cell_counts[0]: # e.g. 100 or 200
            if total_cells == 100: # 50+50 configuration
                df_cells = df[df['Score No.'] == 50].copy()
            else:  # 200 (from 100+100 configuration)
                df_cells = df[df['Score No.'] == 100].copy()
        else: # e.g. 200 or 500 (second count)
            if total_cells == 200: # From 50+50 configuration, so sum of two scores
                df_cells = df[df['Score No.'].isin([50, 100])].copy() # Should this be sum or just specific scores?
                                                                    # Original code implies different Score No. for different techs or stages
                                                                    # For now, assuming Score No. directly relates to one tech's count stage
                                                                    # If 200 cells are from 50 (Tech1) + 50 (Tech2) + 50 (Tech1_Review) + 50 (Tech2_Review)
                                                                    # This logic might need refinement based on exact meaning of 'Score No.' sums.
                                                                    # The current logic implies Score No. 50 is first part, 100 is second part (for 100/200 setup)
                                                                    # And Score No. 100 is first, 150 is second (for 200/500 setup)
                                                                    # Let's stick to original GUI's implied logic:
                df_cells = df[df['Score No.'] == 100].copy() # If target is 200 (second count for 100/200 setup)
                                                            # This means 2 * 50 from Tech1 and 2 * 50 from Tech2 are summed up if Score No 50 + 100 are present
                                                            # The logic for sum comes from how Value is aggregated, not Score No. filter here.

            else:  # 500 (from 100+150 configuration)
                df_cells = df[df['Score No.'].isin([100,150])].copy() # Sum of scores for 500

        # Get unique cases and patterns
        cases = df_cells['Case No.'].unique()
        all_patterns_short = df_cells['Signal_Pattern_Short'].unique()
        
        patterns_to_process = [p for p in all_patterns_short if p in self.selected_patterns]
        self.log_message(f"Processing {len(patterns_to_process)} selected patterns for {total_cells} cells")

        result_columns = ['Case No.', 'Hybe date', 'Case ID', '# Scored']
        # Get unique case info, drop duplicates to avoid issues if a case has multiple rows for the same hybe date/ID
        case_info_df = df_cells[['Case No.', 'Hybe date', 'Case ID']].drop_duplicates(subset=['Case No.'])
        result_df = pd.DataFrame(case_info_df).reset_index(drop=True)

        if 'Hybe date' in result_df.columns:
            result_df['Hybe date'] = pd.to_datetime(result_df['Hybe date'], errors='coerce').dt.strftime('%m/%d/%y')
        result_df['# Scored'] = total_cells

        for pattern in patterns_to_process:
            tech1_col = f"{pattern}_Tech 1"
            tech2_col = f"{pattern}_Tech 2"
            pct_col = f"{pattern}_%"

            result_df[tech1_col] = 0
            result_df[tech2_col] = 0
            result_df[pct_col] = 0.0

            for idx, case_row in result_df.iterrows():
                case_no_current = case_row['Case No.']
                
                # Tech 1 data for the pattern and case
                tech1_data = df_cells[(df_cells['Case No.'] == case_no_current) &
                                     (df_cells['Signal_Pattern_Short'] == pattern) &
                                     (df_cells['Tech No.'] == 'Tech 1')]
                tech1_value = tech1_data['Value'].fillna(0).sum() # Sum if multiple scores contribute (e.g. for 200/500 totals)

                # Tech 2 data
                tech2_data = df_cells[(df_cells['Case No.'] == case_no_current) &
                                     (df_cells['Signal_Pattern_Short'] == pattern) &
                                     (df_cells['Tech No.'] == 'Tech 2')]
                tech2_value = tech2_data['Value'].fillna(0).sum()

                sum_value = tech1_value + tech2_value
                pct_value = (sum_value / total_cells) * 100 if total_cells > 0 else 0

                result_df.loc[idx, tech1_col] = tech1_value
                result_df.loc[idx, tech2_col] = tech2_value
                result_df.loc[idx, pct_col] = pct_value
        
        # Handle combined FRG>1swa (Break Apart)
        frg3_patterns_selected = [p for p in patterns_to_process if 'FRG>3swa' in p or 'FRW>3swa' in p]
        frg1_3_patterns_selected = [p for p in patterns_to_process if any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa'])]

        if frg3_patterns_selected and frg1_3_patterns_selected:
            self.log_message("Creating combined FRG>1swa pattern (Break Apart)...", "blue")
            # Assume only one of each type will be practically selected or the first one found
            frg3_p_name = frg3_patterns_selected[0]
            frg1_3_p_name = frg1_3_patterns_selected[0]

            combined_tech1_col = "FRG>1swa_Tech 1" # Standardized combined name
            combined_tech2_col = "FRG>1swa_Tech 2"
            combined_pct_col = "FRG>1swa_%"

            result_df[combined_tech1_col] = result_df.get(f"{frg3_p_name}_Tech 1", 0) + result_df.get(f"{frg1_3_p_name}_Tech 1", 0)
            result_df[combined_tech2_col] = result_df.get(f"{frg3_p_name}_Tech 2", 0) + result_df.get(f"{frg1_3_p_name}_Tech 2", 0)
            
            combined_sum_values = result_df[combined_tech1_col] + result_df[combined_tech2_col]
            result_df[combined_pct_col] = (combined_sum_values / total_cells) * 100 if total_cells > 0 else 0
            
            # Add "FRG>1swa" to patterns_to_process if it's not there for the HTML table generation to pick it up
            if "FRG>1swa" not in patterns_to_process:
                 # This modification should ideally happen before HTML table generation logic or
                 # HTML table generation should be aware of this dynamic addition.
                 # For now, the create_table_content will pick up any column ending in _%
                 pass


        return result_df


    def calculate_cutoff_and_grey_zones(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Beta method) for {total_cells} cells...")
        pattern_names = set()
        for col in df.columns:
            if '_%' in col: pattern_names.add(col.split('_')[0])
        results = []
        for pattern in pattern_names:
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if tech1_col in df.columns and tech2_col in df.columns and pct_col in df.columns:
                sum_values = df[tech1_col] + df[tech2_col]
                max_sum = sum_values.max()
                percentages = df[pct_col].values / 100
                percentages = percentages[~np.isnan(percentages)]
                if len(percentages) > 0:
                    percentages_for_sd = percentages.copy()
                    if np.all(percentages_for_sd == 0):
                        self.log_message(f"WARNING: All percentage values are 0 for pattern {pattern}. Adding 0.5% to one case for SD calculation (Beta).", "orange")
                        if len(percentages_for_sd) > 0: percentages_for_sd[0] = 0.005
                    
                    n, k = total_cells, min(int(max_sum if pd.notna(max_sum) else 0), total_cells -1) # ensure k < n
                    if k < 0: k = 0 # ensure k is not negative

                    confidence_level = 0.95
                    # Ensure k+1 and n-k parameters are > 0 for beta.ppf
                    # alpha = k+1, beta_param = n-k. Original was n-k-1.
                    # For beta distribution parameters (a,b), a=events+1, b=trials-events+1
                    # So a = k+1, b = n-k+1. Scipy beta.ppf(q, a, b)
                    # Original code was beta.ppf(confidence_level, k+1, n-k-1)
                    # This corresponds to a = k+1, b = (n-1)-k
                    # Let's use the definition where a = number of successes + 1, b = number of failures + 1
                    # So k_successes = k, n_failures = n-k. Then a = k+1, b = n-k+1.
                    # If k=n, then n-k+1 = 1. If k=0, k+1=1.
                    # Beta parameters must be positive.
                    param_a = k + 1
                    param_b = total_cells - k + 1 # Corrected based on common beta usage for proportions
                    if param_a <=0 : param_a = 1e-9 # small positive
                    if param_b <=0 : param_b = 1e-9 # small positive

                    cutoff_beta = beta.ppf(confidence_level, param_a, param_b) if k <= total_cells else 1.0
                    cutoff_beta_pct = cutoff_beta * 100
                    
                    std_dev = np.std(percentages_for_sd) * 100
                    if std_dev < 0.1:
                        std_dev = 0.5
                        self.log_message(f"Applied minimum SD of 0.5% for pattern {pattern} (Beta method)", "orange")
                    
                    grey_zone_lower = max(0, cutoff_beta_pct - 2 * std_dev)
                    grey_zone_upper = min(100, cutoff_beta_pct + 2 * std_dev)
                    non_zero_values = sum_values.values[sum_values.values >= 0] # Should be >= 0
                    data_range = f"{int(min(non_zero_values))}-{int(max(non_zero_values))}" if len(non_zero_values) > 0 else "0-0"
                    
                    results.append({
                        'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_beta_pct:.2f}%",
                        'Grey_Zone_Lower': f"{grey_zone_lower:.2f}%", 'Grey_Zone_Upper': f"{grey_zone_upper:.2f}%",
                        'Range': data_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"
                    })
        self.log_message(f"Calculated {len(results)} Beta cutoff values for {total_cells} cells")
        return pd.DataFrame(results)

    def calculate_cutoff_and_grey_zones_cribinom(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Cribinom method) for {total_cells} cells...")
        pattern_names = set()
        for col in df.columns:
            if '_%' in col: pattern_names.add(col.split('_')[0])
        results = []
        confidence_level = 0.95
        for pattern in pattern_names:
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if tech1_col in df.columns and tech2_col in df.columns and pct_col in df.columns:
                sum_values = df[tech1_col] + df[tech2_col]
                percentages = df[pct_col].values / 100
                percentages = percentages[~np.isnan(percentages)]
                if len(percentages) > 0:
                    p_for_cutoff = np.mean(percentages)
                    if np.all(percentages == 0) or p_for_cutoff == 0:
                        p_for_cutoff = 0.005 
                        self.log_message(f"WARNING: All percentages are 0 or mean is 0 for pattern '{pattern}'. Using p=0.005 for Cribinom.", "orange")

                    # binom.ppf(q, n, p) -> finds k such that P(X <= k) >= q
                    cutoff_count = binom.ppf(confidence_level, n=total_cells, p=p_for_cutoff)
                    if cutoff_count == 0 and p_for_cutoff < 0.01: # Check added from original
                         cutoff_count = 1 # Ensure non-zero cutoff for very low p
                         self.log_message(f"Adjusted Cribinom cutoff for '{pattern}' from 0 to 1.", "orange")

                    cutoff_cribinom_pct = (cutoff_count / total_cells) * 100
                    grey_zone_lower = cutoff_cribinom_pct
                    grey_zone_upper = min(100, cutoff_cribinom_pct + 3.0)
                    
                    std_dev = np.std(percentages) * 100
                    if std_dev < 0.1 and np.any(percentages > 0):
                        std_dev = 0.5
                        self.log_message(f"Applied minimum SD of 0.5% for pattern {pattern} (Cribinom method)", "orange")
                    elif np.all(percentages == 0): std_dev = 0.0
                        
                    non_zero_values = sum_values.values[sum_values.values >= 0]
                    data_range = f"{int(min(non_zero_values))}-{int(max(non_zero_values))}" if len(non_zero_values) > 0 else "0-0"
                    results.append({
                        'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_cribinom_pct:.2f}%",
                        'Grey_Zone_Lower': f"{grey_zone_lower:.2f}%", 'Grey_Zone_Upper': f"{grey_zone_upper:.2f}%",
                        'Range': data_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"
                    })
        self.log_message(f"Calculated {len(results)} Cribinom cutoff values for {total_cells} cells")
        return pd.DataFrame(results)

    def create_reorganized_html_content(self, df_first, df_second):
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

    def create_table_content(self, df, total_cells):
        if df.empty: return "<p>No data available</p>"
        base_cols = ['Case No.', 'Hybe date', 'Case ID', '# Scored']
        # Dynamically find all pattern columns based on the % suffix
        pattern_names = sorted(list(set(col.split('_')[0] for col in df.columns if col.endswith('_Tech 1') or col.endswith('_Tech 2') or col.endswith('_'))))
        
        # Filter pattern_names to include only those derived from self.selected_patterns OR the combined FRG>1swa
        # First, get the short names of selected patterns
        selected_short_patterns = []
        for sp in self.selected_patterns: # self.selected_patterns are short names already
            selected_short_patterns.append(sp)

        # Check if combined FRG needs to be added
        frg3_selected = any('FRG>3swa' in p or 'FRW>3swa' in p for p in selected_short_patterns)
        frg1_3_selected = any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa'] for p in selected_short_patterns)

        final_pattern_names_for_table = []
        for pn in pattern_names: # pn are short names like '1F1G'
            if pn in selected_short_patterns:
                final_pattern_names_for_table.append(pn)
        
        if frg3_selected and frg1_3_selected and "FRG>1swa" in pattern_names and "FRG>1swa" not in final_pattern_names_for_table :
            final_pattern_names_for_table.append("FRG>1swa") # Add the combined pattern if generated
        
        final_pattern_names_for_table = sorted(list(set(final_pattern_names_for_table))) # Ensure unique and sorted


        max_values, max_row_indices = {}, {}
        for pattern in final_pattern_names_for_table:
            pct_col = f"{pattern}_%"
            if pct_col in df.columns:
                max_val = df[pct_col].max()
                if pd.notna(max_val) and max_val > 0: # Check for NaN and ensure >0
                    max_values[pattern] = max_val
                    # Get the first index if multiple rows have the max value
                    max_row_indices[pattern] = df[df[pct_col] == max_val].index[0] 
        html = "<table><thead><tr>"
        for col in base_cols: html += f'<th rowspan="2" class="header-light-blue">{col}</th>'
        for pattern in final_pattern_names_for_table: html += f'<th colspan="3" class="header-dark-blue">{pattern}</th>'
        html += '</tr><tr>'
        for _ in final_pattern_names_for_table: html += '<th class="header-dark-blue">Tech 1</th><th class="header-dark-blue">Tech 2</th><th class="header-dark-blue">%</th>'
        html += '</tr></thead><tbody>'
        for row_idx, row_data in df.iterrows(): # Use df.iterrows() to get actual DataFrame row index
            html += '<tr class="data-row">'
            for col in base_cols: html += f'<td>{row_data.get(col, "")}</td>'
            for pattern in final_pattern_names_for_table:
                t1, t2, pct = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
                v1, v2, vp = row_data.get(t1, 0), row_data.get(t2, 0), row_data.get(pct, 0.0)
                highlight_class = "highlight" if pattern in max_values and max_values.get(pattern) == vp and row_idx == max_row_indices.get(pattern) and vp > 0 else ""
                html += f'<td class="{highlight_class}">{v1}</td><td class="{highlight_class}">{v2}</td><td class="{highlight_class}">{vp:.2f}%</td>'
            html += '</tr>'
        html += "</tbody></table>"
        return html

    def create_cutoff_html_content(self, cutoff_first, cutoff_second):
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

    def create_compact_pattern_sections(self, df_results):
        html = ""
        for _, row in df_results.iterrows():
            html += f"""<div class="pattern-container"><div class="pattern-header">{row['Signal_Pattern']}</div>
                <table class="compact-table">
                <tr><td class="label-cell">Cutoff (95% C.I.)</td><td class="value-cell cutoff-value">{row['Cutoff_95%_CI']}</td></tr>
                <tr><td class="label-cell">Gray Zone</td><td class="value-cell grey-zone">{row['Grey_Zone_Lower']} to {row['Grey_Zone_Upper']}</td></tr>
                <tr><td class="label-cell">Range</td><td class="value-cell range-value">{row['Range']}</td></tr>
                <tr><td class="label-cell">Std. Dev.</td><td class="value-cell std-value">{row['Standard_Deviation']} <span class="controls">{row['Controls']}</span></td></tr>
                </table></div>"""
        return html

# === Helper Functions (Adapted from FISHAnalysisGUI Tkinter Class) ===
def streamlit_log_message(message, color="black", bold=False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    style = ""
    if bold: style += "font-weight:bold;"
    if color != "black": style += f"color:{color};"

    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    # Use Markdown for rich text in log
    if style:
        # Basic color support with markdown
        if color == "red": log_entry = f":red[**{log_entry}**]" if bold else f":red[{log_entry}]"
        elif color == "green": log_entry = f":green[**{log_entry}**]" if bold else f":green[{log_entry}]"
        elif color == "blue": log_entry = f":blue[**{log_entry}**]" if bold else f":blue[{log_entry}]"
        elif color == "orange": log_entry = f":orange[**{log_entry}**]" if bold else f":orange[{log_entry}]"
        elif bold: log_entry = f"**{log_entry}**"
    st.session_state.log_messages.append(log_entry)

def process_single_table_excel(df_raw, start, end, log_func):
    try:
        header_start = max(0, start - 1) # Row above "Case No." for merged headers
        table_with_headers = df_raw.iloc[header_start:end+1].copy().reset_index(drop=True)

        if table_with_headers.shape[0] < 2: # Need at least merged_header_row and specific_header_row
            log_func(f"Table at original rows {start+1}-{end+1} is too short for header processing.", "orange")
            return []

        merged_header_row = table_with_headers.iloc[0].tolist()
        specific_header_row = table_with_headers.iloc[1].tolist() # This is the "Case No." row

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
            else: # Fallback if specific header is empty
                combined_headers.append(f"UnnamedCol_{i}")

        # Data table itself starts from 'specific_header_row' which becomes the header
        table_data_rows = df_raw.iloc[start+1:end+1].copy() # Data rows under "Case No."
        table_header_row_df = df_raw.iloc[start:start+1].copy() # The "Case No." row to be used as base headers

        current_headers = table_header_row_df.iloc[0].tolist()
        final_headers = []

        for i, current_header_val in enumerate(current_headers):
            ch_str = str(current_header_val).strip()
            # Use combined_header if it's a pattern column, otherwise use original
            if i < len(combined_headers) and ("Signal patterns" in combined_headers[i] or "Other" in combined_headers[i]):
                final_headers.append(combined_headers[i])
            elif ch_str : # Use original if it's not empty
                final_headers.append(ch_str)
            else: # Fallback if original is also empty
                final_headers.append(f"UnnamedCol_{i}")
        
        # Ensure final_headers has same length as table_data_rows.columns
        if len(final_headers) != table_data_rows.shape[1]:
            log_func(f"Header length mismatch in table at original rows {start+1}-{end+1}. Expected {table_data_rows.shape[1]}, got {len(final_headers)}. Adjusting.", "orange")
            # Truncate or pad final_headers. For now, let pandas handle it or error out if critical.
            # This usually means a malformed Excel table.
            final_headers = final_headers[:table_data_rows.shape[1]] # Truncate if longer
            while len(final_headers) < table_data_rows.shape[1]: # Pad if shorter
                final_headers.append(f"AutoHeader_{len(final_headers)}")


        table_data_rows.columns = final_headers
        table_data_rows = table_data_rows.reset_index(drop=True)

        tech_cols_fill = [col for col in ["Tech No.", "Tech Initial"] if col in table_data_rows.columns]
        if tech_cols_fill:
            table_data_rows[tech_cols_fill] = table_data_rows[tech_cols_fill].ffill()

        # Extract common info, assuming it's consistent or take first valid
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
                            if sig_col_name in score_row: # Check if pattern column exists
                                value = score_row[sig_col_name] if pd.notna(score_row[sig_col_name]) else 0
                                data_point = {
                                    "Case No.": case_no, "Hybe date": hybe_date, "Case ID": case_id,
                                    "Tech No.": tech_num, "Tech Initial": tech_initial,
                                    "Score No.": score_no_val, "Score Sum": score_sum_val,
                                    "Signal_Pattern": sig_col_name, "Value": value
                                }
                                processed_data_list.append(pd.DataFrame([data_point]))
        else:
            log_func(f"Warning: 'Tech No.' column not found in table at original rows {start+1}-{end+1}. Cannot process tech-specific data for this table.", "orange")
        return processed_data_list
    except Exception as e:
        log_func(f"CRITICAL error processing table at original rows {start+1}-{end+1}: {e}", "red", True)
        import traceback
        log_func(traceback.format_exc(),"red") # log stack trace
        return []


def is_valid_table_excel(df_raw, start, end, log_func):
    try:
        table_segment = df_raw.iloc[start:end+1]
        data_rows = table_segment.iloc[1:] # Skip the "Case No." header row for validation content
        if data_rows.empty: return False
        
        valid_value_count, total_cells = 0, 0
        for col_idx in data_rows.columns:
            for value in data_rows[col_idx]:
                total_cells += 1
                if pd.notna(value) and str(value).strip() != "":
                    if isinstance(value, (int, float)) and value != 0:
                        valid_value_count += 1
                    elif not isinstance(value, (int, float)): # Non-numeric, non-empty string
                         valid_value_count += 1
        
        if total_cells == 0: return False
        return (valid_value_count / total_cells) >= 0.05 # At least 5% valid data
    except Exception as e:
        log_func(f"Error during table validation (rows {start+1}-{end+1}): {e}", "red")
        return True # Be permissive if validation itself fails

def read_fish_data_from_excel(uploaded_file_obj, log_func, progress_bar_st):
    try:
        log_func("Reading Excel sheet...")
        df_raw = pd.read_excel(uploaded_file_obj, sheet_name=0, header=None)
        
        table_starts = df_raw[df_raw.iloc[:, 0] == "Case No."].index.tolist() # Check first column for "Case No."
        if not table_starts:
            log_func("No tables found with 'Case No.' in the first column. Ensure Excel format is correct.", "red", True)
            return pd.DataFrame()

        table_ends = table_starts[1:] + [len(df_raw)] # End of df for the last table
        table_ends = [end - 1 for end in table_ends] # end is the last row index of the table

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
                log_func(f"Table {i+1} (original rows {start_idx+1}-{end_idx+1}) excluded due to insufficient valid data.", "orange")
        
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
        raise # Re-raise to be caught by Streamlit button logic

def detect_cell_counts_streamlit(fish_data_df, log_func):
    if fish_data_df is None or fish_data_df.empty: return [200, 500] # Default
    score_nos = fish_data_df['Score No.'].dropna().unique()
    log_func(f"Found Score No. values: {sorted(list(score_nos))}")
    
    counts = [200, 500] # Default
    if 50 in score_nos: # This means 50 cells per tech initially
        counts = [100, 200] # Total 100 (50+50), then total 200 if rescored
    elif 100 in score_nos and 150 in score_nos: # 100 cells per tech, then 150 more
        counts = [200, 500] # Total 200 (100+100), then total 500 if rescored (100+100 + 150+150 -> no, (100+150)*2 implies first tech counts 100, second tech 100. Then for 500, tech1 100+150, tech2 100+150 )
                                # The original logic was: (100+100)=200, (100+150)*2=500. This implies the score numbers are partial counts.
                                # So, if 100 and 150 are present, it implies the 200/500 scheme.
    elif 100 in score_nos: # Only 100 found, default to 200/500 as per original logic
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
    # Selected patterns are already short names
    frg3 = any('FRG>3swa' in p or 'FRW>3swa' in p for p in selected_patterns_short)
    frg1_3 = any(any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for p in selected_patterns_short)
    if frg3 and frg1_3:
        msg = "Break_apart probe patterns selected. Software will automatically generate combined 'FRG>1swa' results."
        st.info(msg)
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
        ('cell_counts', [200, 500]), ('probe_name', ""), ('selected_patterns_ui', []),
        ('data_loaded_successfully', False), ('last_uploaded_filename', None),
        ('current_probe_name_in_widget', "") # To manage probe name input field
    ]:
        if key not in st.session_state: st.session_state[key] = default_val

    # --- Log Display Area (Sidebar or Expander) ---
    with st.sidebar:
        st.header("📋 Processing Log")
        log_placeholder = st.empty()
        if st.button("Clear Log", key="clear_log_sidebar"):
            st.session_state.log_messages = []
            streamlit_log_message("Log cleared by user.")


    # --- Step 1: File Upload and Setup ---
    st.header("Step 1: File Upload and Probe Setup")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx, .xls)", type=["xlsx", "xls"], key="file_uploader_widget")

    # Auto-infer probe name or use user's input
    if uploaded_file:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.data_loaded_successfully = False # Reset flags
            st.session_state.fish_data = None
            st.session_state.available_patterns = []
            st.session_state.selected_patterns_ui = [] # Reset selections on new file

            filename_stem = Path(uploaded_file.name).stem
            potential_probe = filename_stem.split('_')[0] if '_' in filename_stem else \
                              filename_stem.split('-')[0] if '-' in filename_stem else filename_stem
            st.session_state.current_probe_name_in_widget = potential_probe.upper()
            streamlit_log_message(f"Inferred probe name: {st.session_state.current_probe_name_in_widget}", "blue")
    
    # Probe name input reflects changes
    st.session_state.probe_name = st.text_input(
        "Probe Name (e.g., HER2, ALK)",
        value=st.session_state.current_probe_name_in_widget,
        key="probe_name_text_input",
        on_change=lambda: setattr(st.session_state, 'current_probe_name_in_widget', st.session_state.probe_name_text_input.upper().strip())
    ).upper().strip()


    if st.button("Load and Process Data", key="load_data_main_button"):
        st.session_state.log_messages = [] # Clear log for new processing run
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
                        st.warning("⚠️ No signal patterns were identified in the data. Check file content and format.")
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
        c1, c2 = st.columns(2)
        if c1.button("Select All Patterns", key="select_all_btn"):
            st.session_state.selected_patterns_ui = st.session_state.available_patterns[:]
            streamlit_log_message("Selected all available patterns.", "blue")
        if c2.button("Clear All Selections", key="clear_all_btn"):
            st.session_state.selected_patterns_ui = []
            streamlit_log_message("Cleared all pattern selections.", "blue")
        
        # Filter default selections to ensure they are valid options
        valid_defaults = [p for p in st.session_state.selected_patterns_ui if p in st.session_state.available_patterns]

        st.session_state.selected_patterns_ui = st.multiselect(
            "Select Signal Patterns:",
            options=st.session_state.available_patterns,
            default=valid_defaults, # Use the possibly updated list
            key="pattern_multiselect_main"
        )
    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot select patterns.")
    else:
        st.info("Load data in Step 1 to enable pattern selection.")


    # --- Step 3: Generate Reports ---
    st.header("Step 3: Generate Reports")
    if st.session_state.data_loaded_successfully and st.session_state.available_patterns:
        if not st.session_state.selected_patterns_ui:
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
                streamlit_log_message(f"Generating reports for: {st.session_state.selected_patterns_ui}, Probe: {st.session_state.probe_name}, Method: {calc_method_actual}", "blue", True)
                check_frg_pattern_combination_streamlit(st.session_state.selected_patterns_ui, streamlit_log_message)
                
                report_progress_bar = st.progress(0.0, text="Initializing report generation...")
                try:
                    with st.spinner("Generating reports... This may take a few moments."):
                        analyzer = FISHDataAnalyzer(
                            st.session_state.fish_data, st.session_state.cell_counts, 
                            st.session_state.selected_patterns_ui, st.session_state.probe_name,
                            calc_method_actual, streamlit_log_message
                        )
                        
                        report_progress_bar.progress(0.1, text="Generating reorganized data...")
                        df_first_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[0])
                        df_second_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[1])
                        html_reorganized = analyzer.create_reorganized_html_content(df_first_reorg, df_second_reorg)
                        fname_reorg = f"{analyzer.probe_name}_reorganized_data_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"
                        
                        report_progress_bar.progress(0.5, text="Generating cutoff values data...")
                        if calc_method_actual == "Beta":
                            cut_first = analyzer.calculate_cutoff_and_grey_zones(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones(df_second_reorg, analyzer.cell_counts[1])
                        else: # Cribinom
                            cut_first = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_second_reorg, analyzer.cell_counts[1])
                        html_cutoff = analyzer.create_cutoff_html_content(cut_first, cut_second)
                        fname_cutoff = f"{analyzer.probe_name}_{calc_method_actual.lower()}_cutoff_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"
                        
                        report_progress_bar.progress(1.0, text="Reports generated!")
                        st.success("✅ Reports generated successfully!")
                        
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            st.download_button(f"📥 Download Reorganized Data ({fname_reorg})", html_reorganized, fname_reorg, "text/html", key="dl_reorg")
                        with dl_col2:
                            st.download_button(f"📥 Download Cutoff Values ({fname_cutoff})", html_cutoff, fname_cutoff, "text/html", key="dl_cutoff")
                        
                except Exception as e:
                    st.error(f"💥 Failed to generate reports: {e}")
                    streamlit_log_message(f"CRITICAL REPORTING ERROR: {e}", "red", True)
                    import traceback
                    streamlit_log_message(traceback.format_exc(),"red")
                finally:
                   if 'report_progress_bar' in locals(): report_progress_bar.empty()
    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot generate reports.")
    else:
        st.info("Load data and select patterns in Steps 1 & 2 to enable report generation.")

    # Update log display (sidebar)
    log_output = "\n".join(st.session_state.log_messages)
    log_placeholder.markdown(f"```\n{log_output}\n```" if log_output else "No log messages yet.")


if __name__ == "__main__":
    run_fish_analysis_app()
