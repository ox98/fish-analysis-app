import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import beta, binom
import os # Not strictly needed for Streamlit cloud but good for path handling if run locally
from pathlib import Path
from datetime import datetime
import traceback # For logging stack traces

# --- FISHDataAnalyzer Class (Modified to collect critical warnings) ---
class FISHDataAnalyzer:
    def __init__(self, fish_data, cell_counts, selected_patterns, probe_name, calculation_method, log_callback=None):
        self.fish_data = fish_data
        self.cell_counts = cell_counts
        self.selected_patterns = selected_patterns
        self.probe_name = probe_name
        self.calculation_method = calculation_method
        self.log_callback = log_callback
        self.critical_data_warnings = [] # To store specific warnings for prominent display

    def log_message(self, message, color="black", bold=False, is_critical_warning=False):
        """Log message and collect critical warnings."""
        if self.log_callback:
            self.log_callback(message, color, bold)
        if is_critical_warning:
            # Avoid duplicate warnings if the exact same message is generated multiple times for the same reason
            if message not in self.critical_data_warnings:
                self.critical_data_warnings.append(message)

    def reorganize_data_with_techs(self, total_cells):
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
        if total_cells == self.cell_counts[0]:
            if total_cells == 100: df_cells = df[df['Score No.'] == 50].copy()
            else: df_cells = df[df['Score No.'] == 100].copy()
        else:
            if total_cells == 200: df_cells = df[df['Score No.'].isin([50, 100])].copy() # Original used isin for 200/500
            else: df_cells = df[df['Score No.'].isin([100, 150])].copy()

        cases = df_cells['Case No.'].unique()
        all_patterns_short = df_cells['Signal_Pattern_Short'].unique()
        patterns_to_process = [p for p in all_patterns_short if p in self.selected_patterns]
        self.log_message(f"Processing {len(patterns_to_process)} selected patterns for {total_cells} cells")

        case_info_df = df_cells[['Case No.', 'Hybe date', 'Case ID']].drop_duplicates(subset=['Case No.'])
        result_df = pd.DataFrame(case_info_df).reset_index(drop=True)
        if 'Hybe date' in result_df.columns:
            result_df['Hybe date'] = pd.to_datetime(result_df['Hybe date'], errors='coerce').dt.strftime('%m/%d/%y')
        result_df['# Scored'] = total_cells

        for pattern in patterns_to_process:
            tech1_col, tech2_col, pct_col = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            result_df[tech1_col], result_df[tech2_col], result_df[pct_col] = 0, 0, 0.0
            for idx, case_row in result_df.iterrows():
                case_no_current = case_row['Case No.']
                tech1_data = df_cells[(df_cells['Case No.'] == case_no_current) & (df_cells['Signal_Pattern_Short'] == pattern) & (df_cells['Tech No.'] == 'Tech 1')]
                tech1_value = tech1_data['Value'].fillna(0).sum()
                tech2_data = df_cells[(df_cells['Case No.'] == case_no_current) & (df_cells['Signal_Pattern_Short'] == pattern) & (df_cells['Tech No.'] == 'Tech 2')]
                tech2_value = tech2_data['Value'].fillna(0).sum()
                sum_value = tech1_value + tech2_value
                pct_value = (sum_value / total_cells) * 100 if total_cells > 0 else 0
                result_df.loc[idx, tech1_col], result_df.loc[idx, tech2_col], result_df.loc[idx, pct_col] = tech1_value, tech2_value, pct_value
        
        frg3_sel = [p for p in patterns_to_process if 'FRG>3swa' in p or 'FRW>3swa' in p]
        frg1_3_sel = [p for p in patterns_to_process if any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa'])]
        if frg3_sel and frg1_3_sel:
            self.log_message("Creating combined FRG>1swa pattern (Break Apart)...", "blue")
            frg3_p, frg1_3_p = frg3_sel[0], frg1_3_sel[0]
            ct1_col, ct2_col, cp_col = "FRG>1swa_Tech 1", "FRG>1swa_Tech 2", "FRG>1swa_%"
            result_df[ct1_col] = result_df.get(f"{frg3_p}_Tech 1", 0) + result_df.get(f"{frg1_3_p}_Tech 1", 0)
            result_df[ct2_col] = result_df.get(f"{frg3_p}_Tech 2", 0) + result_df.get(f"{frg1_3_p}_Tech 2", 0)
            c_sum_val = result_df[ct1_col] + result_df[ct2_col]
            result_df[cp_col] = (c_sum_val / total_cells) * 100 if total_cells > 0 else 0
        return result_df

    def calculate_cutoff_and_grey_zones(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Beta method) for {total_cells} cells...")
        pattern_names = set(col.split('_')[0] for col in df.columns if '_%' in col)
        results = []
        for pattern in pattern_names:
            t1c, t2c, pc = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if t1c in df.columns and t2c in df.columns and pc in df.columns:
                sum_v = df[t1c] + df[t2c]
                max_s = sum_v.max()
                percentages = df[pc].values / 100
                percentages = percentages[~np.isnan(percentages)]
                if len(percentages) > 0:
                    sd_pcts = percentages.copy()
                    warning_msg_sd = f"Pattern '{pattern}': All percentage values are 0. Adding 0.5% to one case for SD calculation (Beta method)."
                    if np.all(sd_pcts == 0):
                        self.log_message(warning_msg_sd, "orange", is_critical_warning=True)
                        if len(sd_pcts) > 0: sd_pcts[0] = 0.005
                    
                    n, k_val = total_cells, min(int(max_s if pd.notna(max_s) else 0), n - 1)
                    k_val = max(0, k_val)
                    param_a, param_b = k_val + 1, n - k_val + 1
                    if param_a <= 0: param_a = 1e-9
                    if param_b <= 0: param_b = 1e-9
                    cutoff_b = beta.ppf(0.95, param_a, param_b) if k_val <= n else 1.0
                    cutoff_b_pct = cutoff_b * 100
                    
                    std_dev = np.std(sd_pcts) * 100
                    warning_msg_min_sd = f"Pattern '{pattern}': Calculated SD is very low (<0.1%). Applied minimum SD of 0.5% (Beta method)."
                    if std_dev < 0.1:
                        self.log_message(warning_msg_min_sd, "orange", is_critical_warning=True)
                        std_dev = 0.5
                    
                    gz_lower, gz_upper = max(0, cutoff_b_pct - 2 * std_dev), min(100, cutoff_b_pct + 2 * std_dev)
                    non_zero_v = sum_v.values[sum_v.values >= 0]
                    d_range = f"{int(min(non_zero_v))}-{int(max(non_zero_v))}" if len(non_zero_v) > 0 else "0-0"
                    results.append({'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_b_pct:.2f}%",
                                    'Grey_Zone_Lower': f"{gz_lower:.2f}%", 'Grey_Zone_Upper': f"{gz_upper:.2f}%",
                                    'Range': d_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"})
        self.log_message(f"Calculated {len(results)} Beta cutoff values for {total_cells} cells")
        return pd.DataFrame(results)

    def calculate_cutoff_and_grey_zones_cribinom(self, df, total_cells):
        self.log_message(f"Calculating cutoff values (Cribinom method) for {total_cells} cells...")
        pattern_names = set(col.split('_')[0] for col in df.columns if '_%' in col)
        results = []
        for pattern in pattern_names:
            t1c, t2c, pc = f"{pattern}_Tech 1", f"{pattern}_Tech 2", f"{pattern}_%"
            if t1c in df.columns and t2c in df.columns and pc in df.columns:
                sum_v = df[t1c] + df[t2c]
                percentages = df[pc].values / 100
                percentages = percentages[~np.isnan(percentages)]
                if len(percentages) > 0:
                    p_cutoff = np.mean(percentages)
                    warning_msg_crib_p0 = f"Pattern '{pattern}': All percentages are 0 or mean is 0. Using p=0.005 for Cribinom cutoff calculation."
                    if np.all(percentages == 0) or p_cutoff == 0:
                        self.log_message(warning_msg_crib_p0, "orange", is_critical_warning=True)
                        p_cutoff = 0.005
                    
                    cutoff_count = binom.ppf(0.95, n=total_cells, p=p_cutoff)
                    warning_msg_crib_adj = f"Pattern '{pattern}': Calculated Cribinom cutoff count is 0. Adjusted to 1 for percentage calculation."
                    if cutoff_count == 0 and p_cutoff < 0.01:
                        self.log_message(warning_msg_crib_adj, "orange", is_critical_warning=True)
                        cutoff_count = 1
                    
                    cutoff_crib_pct = (cutoff_count / total_cells) * 100
                    gz_lower, gz_upper = cutoff_crib_pct, min(100, cutoff_crib_pct + 3.0)
                    std_dev = np.std(percentages) * 100
                    warning_msg_crib_min_sd = f"Pattern '{pattern}': Calculated SD is very low (<0.1%) with non-zero data. Applied minimum SD of 0.5% (Cribinom method)."
                    if std_dev < 0.1 and np.any(percentages > 0):
                        self.log_message(warning_msg_crib_min_sd, "orange", is_critical_warning=True)
                        std_dev = 0.5
                    elif np.all(percentages == 0): std_dev = 0.0
                    
                    non_zero_v = sum_v.values[sum_v.values >= 0]
                    d_range = f"{int(min(non_zero_v))}-{int(max(non_zero_v))}" if len(non_zero_v) > 0 else "0-0"
                    results.append({'Signal_Pattern': pattern, 'Cutoff_95%_CI': f"{cutoff_crib_pct:.2f}%",
                                    'Grey_Zone_Lower': f"{gz_lower:.2f}%", 'Grey_Zone_Upper': f"{gz_upper:.2f}%",
                                    'Range': d_range, 'Standard_Deviation': f"{std_dev:.2f}%", 'Controls': "(20 controls)"})
        self.log_message(f"Calculated {len(results)} Cribinom cutoff values for {total_cells} cells")
        return pd.DataFrame(results)

    def create_reorganized_html_content(self, df_first, df_second):
        # (HTML generation code from previous response - no change needed here for functionality)
        # ... (ensure this function correctly uses self.selected_patterns and handles FRG>1swa if present) ...
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
        # (HTML generation code from previous response - no change needed here for functionality)
        if df.empty: return "<p>No data available</p>"
        base_cols = ['Case No.', 'Hybe date', 'Case ID', '# Scored']
        pattern_names_in_df = sorted(list(set(col.split('_')[0] for col in df.columns if col.endswith('_Tech 1') or col.endswith('_Tech 2') or col.endswith('_'))))
        
        selected_short_patterns = self.selected_patterns[:] # self.selected_patterns are already short names

        frg3_selected = any('FRG>3swa' in p or 'FRW>3swa' in p for p in selected_short_patterns)
        frg1_3_selected = any(any(substr in p for substr in ['FRG>1swa<3swa', 'FRW>1swa<3swa', 'FRG>1<3swa', 'FRW>1<3swa']) for p in selected_short_patterns)

        final_pattern_names_for_table = [pn for pn in pattern_names_in_df if pn in selected_short_patterns]
        
        if frg3_selected and frg1_3_selected and "FRG>1swa" in pattern_names_in_df and "FRG>1swa" not in final_pattern_names_for_table :
            final_pattern_names_for_table.append("FRG>1swa")
        
        final_pattern_names_for_table = sorted(list(set(final_pattern_names_for_table)))

        max_values, max_row_indices = {}, {}
        for pattern in final_pattern_names_for_table:
            pct_col = f"{pattern}_%"
            if pct_col in df.columns and not df[pct_col].empty: # Ensure column exists and is not empty
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
                is_max = pattern in max_values and pd.notna(max_values.get(pattern)) and \
                           abs(max_values.get(pattern) - vp) < 1e-9 and \
                           row_idx == max_row_indices.get(pattern) and vp > 0 # Comparing floats
                highlight_class = "highlight" if is_max else ""
                html += f'<td class="{highlight_class}">{v1}</td><td class="{highlight_class}">{v2}</td><td class="{highlight_class}">{vp:.2f}%</td>'
            html += '</tr>'
        html += "</tbody></table>"
        return html
        
    def create_cutoff_html_content(self, cutoff_first, cutoff_second):
        # (HTML generation code from previous response - no change needed here for functionality)
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
        # (HTML generation code from previous response - no change needed here for functionality)
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
    
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    
    prefix = suffix = ""
    if bold: prefix, suffix = "**", "**"
    
    if color == "red": log_entry = f":red[{prefix}{log_entry}{suffix}]"
    elif color == "green": log_entry = f":green[{prefix}{log_entry}{suffix}]"
    elif color == "blue": log_entry = f":blue[{prefix}{log_entry}{suffix}]"
    elif color == "orange": log_entry = f":orange[{prefix}{log_entry}{suffix}]"
    elif bold: log_entry = f"{prefix}{log_entry}{suffix}"
        
    st.session_state.log_messages.append(log_entry)

def process_single_table_excel(df_raw, start, end, log_func):
    try:
        header_start = max(0, start - 1)
        table_with_headers = df_raw.iloc[header_start:end+1].copy().reset_index(drop=True)
        if table_with_headers.shape[0] < 2:
            log_func(f"Table at original rows {start+1}-{end+1} is too short for header processing.", "orange")
            return []

        merged_header_row = table_with_headers.iloc[0].tolist()
        specific_header_row = table_with_headers.iloc[1].tolist()
        column_categories = {i: str(merged_header_row[i]).strip() if pd.notna(merged_header_row[i]) and str(merged_header_row[i]).strip() else column_categories.get(i-1, "") for i in range(len(merged_header_row))}
        
        combined_headers = []
        for i, specific_header in enumerate(specific_header_row):
            s_h_str = str(specific_header).strip()
            p_cat = column_categories.get(i, "")
            if s_h_str:
                if p_cat and ("Signal patterns" in p_cat or "Other" in p_cat): combined_headers.append(f"{p_cat}_{s_h_str}")
                else: combined_headers.append(s_h_str)
            else: combined_headers.append(f"UnnamedCol_{i}")

        table_data_rows = df_raw.iloc[start+1:end+1].copy()
        base_headers_list = df_raw.iloc[start:start+1].iloc[0].tolist()
        final_headers = []
        for i, current_h_val in enumerate(base_headers_list):
            ch_str = str(current_h_val).strip()
            if i < len(combined_headers) and ("Signal patterns" in combined_headers[i] or "Other" in combined_headers[i]) and not combined_headers[i].startswith("UnnamedCol_") :
                final_headers.append(combined_headers[i])
            elif ch_str: final_headers.append(ch_str)
            else: final_headers.append(f"UnnamedCol_{i}")
        
        if len(final_headers) != table_data_rows.shape[1]:
            log_func(f"Header length mismatch (table rows {start+1}-{end+1}): expected {table_data_rows.shape[1]}, got {len(final_headers)}. Adjusting.", "orange")
            final_headers = final_headers[:table_data_rows.shape[1]]
            while len(final_headers) < table_data_rows.shape[1]: final_headers.append(f"AutoHeader_{len(final_headers)}")
        
        table_data_rows.columns = final_headers
        table_data_rows = table_data_rows.reset_index(drop=True)
        tech_cols = [col for col in ["Tech No.", "Tech Initial"] if col in table_data_rows.columns]
        if tech_cols: table_data_rows[tech_cols] = table_data_rows[tech_cols].ffill()

        case_no = table_data_rows["Case No."].iloc[0] if "Case No." in final_headers and pd.notna(table_data_rows["Case No."].iloc[0]) else None
        hybe_date = table_data_rows["Hybe date"].iloc[0] if "Hybe date" in final_headers and pd.notna(table_data_rows["Hybe date"].iloc[0]) else None
        case_id = table_data_rows["Case ID"].iloc[0] if "Case ID" in final_headers and pd.notna(table_data_rows["Case ID"].iloc[0]) else None
        sig_pattern_cols = [col for col in final_headers if isinstance(col, str) and ("Signal patterns" in col or "Other" in col)]
        
        processed_list = []
        if "Tech No." in final_headers:
            for tech_n, group in table_data_rows.groupby("Tech No."):
                tech_i = group["Tech Initial"].iloc[0] if "Tech Initial" in final_headers and pd.notna(group["Tech Initial"].iloc[0]) else None
                for _, s_row in group.iterrows():
                    if "Score No." in final_headers and pd.notna(s_row["Score No."]):
                        score_n_val, score_s_val = s_row["Score No."], s_row.get("Score Sum", None)
                        for sig_col in sig_pattern_cols:
                            if sig_col in s_row:
                                val = s_row[sig_col] if pd.notna(s_row[sig_col]) else 0
                                processed_list.append(pd.DataFrame([{"Case No.": case_no, "Hybe date": hybe_date, "Case ID": case_id,
                                                                    "Tech No.": tech_n, "Tech Initial": tech_i, "Score No.": score_n_val,
                                                                    "Score Sum": score_s_val, "Signal_Pattern": sig_col, "Value": val}]))
        else:
            log_func(f"Warning: 'Tech No.' column not found in table at original rows {start+1}-{end+1}.", "orange")
        return processed_list
    except Exception as e:
        log_func(f"CRITICAL error processing table at original rows {start+1}-{end+1}: {e}", "red", True)
        log_func(traceback.format_exc(),"red")
        return []

def is_valid_table_excel(df_raw, start, end, log_func):
    # (Function from previous response - no change needed here for functionality)
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
    # (Function from previous response - no change needed here for functionality)
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
                excluded_count += 1; continue

            if is_valid_table_excel(df_raw, start_idx, end_idx, log_func):
                processed_parts = process_single_table_excel(df_raw, start_idx, end_idx, log_func)
                if processed_parts:
                    all_tables_data_list.extend(processed_parts)
                    log_func(f"Table {i+1} (original rows {start_idx+1}-{end_idx+1}) processed.")
            else:
                excluded_count += 1
                log_func(f"Table {i+1} (original rows {start_idx+1}-{end_idx+1}) excluded due to insufficient valid data.", "orange")
        
        if excluded_count > 0: log_func(f"INFO: Excluded {excluded_count} tables.", "blue")
        
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
        log_func(traceback.format_exc(),"red")
        raise

def detect_cell_counts_streamlit(fish_data_df, log_func):
    # (Function from previous response - no change needed here for functionality)
    if fish_data_df is None or fish_data_df.empty: return [200, 500]
    score_nos = fish_data_df['Score No.'].dropna().unique()
    log_func(f"Found Score No. values: {sorted(list(score_nos))}")
    counts = [200, 500]
    if 50 in score_nos: counts = [100, 200]
    elif 100 in score_nos and 150 in score_nos: counts = [200, 500]
    elif 100 in score_nos: counts = [200, 500]
    log_func(f"Determined cell counts: {counts}")
    return counts

def get_available_patterns_streamlit(fish_data_df, log_func):
    # (Function from previous response - no change needed here for functionality)
    if fish_data_df is None or fish_data_df.empty: return []
    unique_patterns_raw = fish_data_df['Signal_Pattern'].unique()
    pattern_map = {p_raw: str(p_raw).split('_')[-1] if '_' in str(p_raw) else str(p_raw) for p_raw in unique_patterns_raw}
    available_short_patterns = sorted(list(set(pattern_map.values())))
    log_func(f"Available short signal patterns: {available_short_patterns}")
    return available_short_patterns

def check_frg_pattern_combination_streamlit(selected_patterns_short, log_func):
    # (Function from previous response - no change needed here for functionality)
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

    # Initialize session state more robustly
    default_session_state = {
        'log_messages': [], 'fish_data': None, 'available_patterns': [],
        'cell_counts': [200, 500], 'probe_name': "", 'selected_patterns_ui': [],
        'data_loaded_successfully': False, 'last_uploaded_filename': None,
        'current_probe_name_in_widget': "",
        'reports_generated': False, # For persistent download links
        'html_reorganized_content': None, 'fname_reorg': "",
        'html_cutoff_content': None, 'fname_cutoff': "",
        'active_report_warnings': [] # For prominent warnings
    }
    for key, default_val in default_session_state.items():
        if key not in st.session_state: st.session_state[key] = default_val

    with st.sidebar:
        st.header("📋 Processing Log")
        log_placeholder = st.empty()
        if st.button("Clear Log", key="clear_log_sidebar_btn"):
            st.session_state.log_messages = ["Log cleared by user."] # Keep one message
            # Do not clear active_report_warnings here as they relate to generated reports

    st.header("Step 1: File Upload and Probe Setup")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx, .xls)", type=["xlsx", "xls"], key="file_uploader_main_widget")

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_uploaded_filename:
            st.session_state.last_uploaded_filename = uploaded_file.name
            st.session_state.data_loaded_successfully = False
            st.session_state.fish_data = None; st.session_state.available_patterns = []
            st.session_state.selected_patterns_ui = []
            st.session_state.reports_generated = False # Reset report state on new file
            st.session_state.active_report_warnings = []

            filename_stem = Path(uploaded_file.name).stem
            potential_probe = filename_stem.split('_')[0] if '_' in filename_stem else \
                              filename_stem.split('-')[0] if '-' in filename_stem else filename_stem
            st.session_state.current_probe_name_in_widget = potential_probe.upper()
            streamlit_log_message(f"Inferred probe name: {st.session_state.current_probe_name_in_widget}", "blue")
    
    # Update probe_name from widget, and widget from probe_name if it changes elsewhere
    if st.session_state.probe_name != st.session_state.current_probe_name_in_widget : # if probe_name was inferred
         st.session_state.current_probe_name_in_widget = st.session_state.probe_name

    st.session_state.probe_name = st.text_input(
        "Probe Name (e.g., HER2, ALK)",
        value=st.session_state.current_probe_name_in_widget,
        key="probe_name_ti_key",
        on_change=lambda: setattr(st.session_state, 'current_probe_name_in_widget', st.session_state.probe_name_ti_key.upper().strip())
    ).upper().strip()


    if st.button("Load and Process Data", key="load_process_data_btn"):
        st.session_state.log_messages = [] # Clear log for new processing run
        st.session_state.reports_generated = False # Reset report state
        st.session_state.active_report_warnings = []
        streamlit_log_message("Attempting to load and process data...")
        if not uploaded_file:
            st.error("❌ Please upload an Excel file first.")
            streamlit_log_message("Error: No Excel file uploaded.", "red", True)
        elif not st.session_state.probe_name:
            st.error("❌ Probe name is required. Please enter a probe name.")
            streamlit_log_message("Error: Probe name missing.", "red", True)
        else:
            # ... (data loading logic from previous response, ensure progress bar is handled)
            streamlit_log_message(f"Processing for probe: {st.session_state.probe_name}", "blue")
            load_prog_bar = st.progress(0.0, text="Initializing data load...")
            try:
                with st.spinner("Reading and analyzing Excel file... This might take a moment."):
                    st.session_state.fish_data = read_fish_data_from_excel(uploaded_file, streamlit_log_message, load_prog_bar)
                
                if st.session_state.fish_data is not None and not st.session_state.fish_data.empty:
                    load_prog_bar.progress(0.75, text="Detecting cell counts and patterns...")
                    st.session_state.cell_counts = detect_cell_counts_streamlit(st.session_state.fish_data, streamlit_log_message)
                    st.session_state.available_patterns = get_available_patterns_streamlit(st.session_state.fish_data, streamlit_log_message)
                    load_prog_bar.progress(1.0, text="Data loaded successfully!")
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
                streamlit_log_message(traceback.format_exc(),"red")
                st.session_state.data_loaded_successfully = False
            finally:
                if 'load_prog_bar' in locals() : load_prog_bar.empty()


    st.header("Step 2: Signal Pattern Selection")
    if st.session_state.data_loaded_successfully and st.session_state.available_patterns:
        # ... (Pattern selection UI from previous response - no change needed here for functionality)
        c1, c2 = st.columns(2)
        if c1.button("Select All Patterns", key="select_all_patterns_btn"):
            st.session_state.selected_patterns_ui = st.session_state.available_patterns[:]
            streamlit_log_message("Selected all available patterns.", "blue")
        if c2.button("Clear All Selections", key="clear_all_patterns_btn"):
            st.session_state.selected_patterns_ui = []
            streamlit_log_message("Cleared all pattern selections.", "blue")
        
        valid_defaults = [p for p in st.session_state.selected_patterns_ui if p in st.session_state.available_patterns]
        if len(valid_defaults) != len(st.session_state.selected_patterns_ui) and st.session_state.selected_patterns_ui:
             streamlit_log_message("Note: Some previously selected patterns are no longer available with the current data and have been unselected.", "orange")
             st.session_state.selected_patterns_ui = valid_defaults


        st.session_state.selected_patterns_ui = st.multiselect(
            "Select Signal Patterns:",
            options=st.session_state.available_patterns,
            default=st.session_state.selected_patterns_ui, 
            key="pattern_multiselect_main_widget"
        )

    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot select patterns.")
    else:
        st.info("Load data in Step 1 to enable pattern selection.")

    st.header("Step 3: Generate Reports")
    if st.session_state.data_loaded_successfully and st.session_state.available_patterns:
        if not st.session_state.selected_patterns_ui:
            st.warning("⚠️ Please select at least one signal pattern in Step 2 to generate reports.")
        else:
            method_label = st.radio(
                "Select Cut-off Calculation Method:",
                ("Beta Inverse Function (Original Method)", "CRITBINOM Function (Alternative Method)"),
                format_func=lambda x: x.split(" (")[0], key="calc_method_radio_widget"
            )
            calc_method_actual = "Beta" if "Beta" in method_label else "Cribinom"

            if st.button("Generate HTML Reports", key="generate_reports_main_btn"):
                st.session_state.active_report_warnings = [] # Clear previous run's warnings
                streamlit_log_message(f"Generating reports for: {st.session_state.selected_patterns_ui}, Probe: {st.session_state.probe_name}, Method: {calc_method_actual}", "blue", True)
                check_frg_pattern_combination_streamlit(st.session_state.selected_patterns_ui, streamlit_log_message)
                
                report_prog_bar = st.progress(0.0, text="Initializing report generation...")
                try:
                    with st.spinner("Generating reports... This may take a few moments."):
                        analyzer = FISHDataAnalyzer(
                            st.session_state.fish_data, st.session_state.cell_counts, 
                            st.session_state.selected_patterns_ui, st.session_state.probe_name,
                            calc_method_actual, streamlit_log_message
                        )
                        
                        report_prog_bar.progress(0.1, text="Generating reorganized data...")
                        df_first_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[0])
                        df_second_reorg = analyzer.reorganize_data_with_techs(analyzer.cell_counts[1])
                        st.session_state.html_reorganized_content = analyzer.create_reorganized_html_content(df_first_reorg, df_second_reorg)
                        st.session_state.fname_reorg = f"{analyzer.probe_name}_reorganized_data_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"
                        
                        report_prog_bar.progress(0.5, text="Generating cutoff values data...")
                        if calc_method_actual == "Beta":
                            cut_first = analyzer.calculate_cutoff_and_grey_zones(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones(df_second_reorg, analyzer.cell_counts[1])
                        else: # Cribinom
                            cut_first = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_first_reorg, analyzer.cell_counts[0])
                            cut_second = analyzer.calculate_cutoff_and_grey_zones_cribinom(df_second_reorg, analyzer.cell_counts[1])
                        st.session_state.html_cutoff_content = analyzer.create_cutoff_html_content(cut_first, cut_second)
                        st.session_state.fname_cutoff = f"{analyzer.probe_name}_{calc_method_actual.lower()}_cutoff_{analyzer.cell_counts[0]}_{analyzer.cell_counts[1]}.html"
                        
                        st.session_state.active_report_warnings = analyzer.critical_data_warnings[:] # Store warnings
                        st.session_state.reports_generated = True # Flag that reports are ready
                        report_prog_bar.progress(1.0, text="Reports generated!")
                        st.success("✅ Reports generated successfully!")
                        
                except Exception as e:
                    st.error(f"💥 Failed to generate reports: {e}")
                    streamlit_log_message(f"CRITICAL REPORTING ERROR: {e}", "red", True)
                    streamlit_log_message(traceback.format_exc(),"red")
                    st.session_state.reports_generated = False # Ensure flag is false on error
                finally:
                   if 'report_prog_bar' in locals(): report_prog_bar.empty()

    elif st.session_state.data_loaded_successfully:
        st.warning("Data loaded, but no signal patterns found. Cannot generate reports.")
    else:
        st.info("Load data and select patterns in Steps 1 & 2 to enable report generation.")

    # Display warnings and download buttons if reports have been generated
    if st.session_state.get('reports_generated', False):
        if st.session_state.get('active_report_warnings'):
            st.subheader("⚠️ Data Processing Notices")
            for warning_msg in st.session_state.active_report_warnings:
                st.warning(warning_msg)
        
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            if st.session_state.get('html_reorganized_content') and st.session_state.get('fname_reorg'):
                st.download_button(
                    label=f"📥 Download Reorganized Data ({st.session_state.fname_reorg})",
                    data=st.session_state.html_reorganized_content,
                    file_name=st.session_state.fname_reorg,
                    mime="text/html",
                    key="download_reorganized_data_btn"
                )
        with dl_col2:
            if st.session_state.get('html_cutoff_content') and st.session_state.get('fname_cutoff'):
                st.download_button(
                    label=f"📥 Download Cutoff Values ({st.session_state.fname_cutoff})",
                    data=st.session_state.html_cutoff_content,
                    file_name=st.session_state.fname_cutoff,
                    mime="text/html",
                    key="download_cutoff_data_btn"
                )

    log_output = "\n".join(st.session_state.log_messages)
    log_placeholder.markdown(f"<pre style='max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 5px; font-size: 0.8em;'>{log_output}</pre>" if log_output else "No log messages yet.", unsafe_allow_html=True)


if __name__ == "__main__":
    run_fish_analysis_app()
