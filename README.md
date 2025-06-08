
FISH Probe Cut-off Calculation Tool🔬


	This is a Streamlit web application designed to analyze Fluorescence In Situ Hybridization (FISH) probe data from Excel files. It processes raw scoring data, reorganizes it by cell count, calculates statistical cut-off values and grey zones using two different methods, and generates comprehensive HTML reports for laboratory use.
	
 Live Application Link
	You can access the live application here:[https://fish-analysis-app-ohz7j7ytrei9pwhkzkgydk.streamlit.app/]


Features
	    • Excel Data Upload: Securely upload and parse complex, multi-table Excel spreadsheets containing FISH scoring data.
	    • Automatic Data Reorganization: Aggregates data from two technicians and reorganizes it based on the number of cells scored (e.g., 100/200 or 200/500 cell counts).
	    • Dynamic Pattern Selection: Automatically detects all signal patterns present in the data and allows the user to select which patterns to include in the analysis, respecting the selection order in the final reports.
	    • Statistical Cut-off Calculation: Implements two distinct methods for determining the normal cut-off value for abnormal signal patterns:
	        ◦ Beta Inverse Function (Original Method)
	        ◦ CRITBINOM Function (Alternative Method)
	    • Grey Zone Determination: Calculates a "grey zone" around the cut-off to aid in interpreting borderline cases.
	    • Report Generation: Produces two downloadable HTML reports:
	        ◦ A detailed table of the reorganized scoring data.
	        ◦ A compact summary of the cut-off, grey zone, and other statistics for each selected pattern.


How It Works

	The application follows a simple three-step process for analysis:
	Step 1: File Upload and Probe Setup
	    1. Upload Excel File: Click the "Browse files" button to upload your .xlsx or .xls data file.
	    2. Enter Probe Name: The app will attempt to infer the probe name from the filename. Confirm or enter the correct probe name (e.g., HER2, ALK).
	    3. Load and Process Data: Click the "Load and Process Data" button. The application will parse the Excel file, identify all unique signal patterns, and determine the cell count schemes (e.g., 100/200 or 200/500) based on the "Score No." columns in your data.
	Step 2: Signal Pattern Selection
	    1. Select Patterns: A multi-select box will appear with all available signal patterns found in your data.
	    2. Set Report Order: Select the patterns you wish to analyze. The order in which you select them is the order they will appear in the final reports.
	    3. Manage Selection: Use the "Select All" and "Clear All" buttons for convenience. Your selection order is displayed below the selection box.
	Step 3: Generate Reports
	    1. Choose Calculation Method: Select either the "Beta Inverse Function" or "CRITBINOM Function" for the cut-off calculation.
	    2. Generate HTML Reports: Click the "Generate HTML Reports" button.
	    3. Download: Download links for the two HTML reports (Reorganized Data and Cutoff Values) will appear.




Methodology for Cut-off and Grey Zone

	The application provides two statistical methods to establish the upper limit of the normal range (cut-off) for signal patterns based on a control dataset (assumed to be from normal cases).

Beta Inverse Function Method

	This method models the expected proportion of abnormal cells using the Beta distribution, which is well-suited for modeling probabilities.
	    1. Inputs:
	        ◦ n: The total number of cells scored (e.g., 100, 200, 500).
	        ◦ k: The maximum number of abnormal cells observed for a specific pattern across all control cases.
	    2. Calculation: The cut-off is calculated using the Beta distribution's inverse cumulative distribution function (also known as the quantile function or ppf). The 95% confidence interval is used to find the proportion (p) at which one can be 95% confident that it represents the upper limit of normal.
	        ◦ Cut-off proportion = Beta.ppf(0.95, a, b)
	        ◦ Where parameters a=k+1 and b=n−k+1.
	    3. Grey Zone: The grey zone is established around the cut-off value to account for statistical variability.
	        ◦ Grey Zone = Cut-off % ± 2 * (Standard Deviation of observed percentages)
CRITBINOM Function Method

	This method uses the binomial distribution to determine the number of abnormal cells that would be critically rare (e.g., less than 5% chance of occurring) in a normal population.
	    1. Inputs:
	        ◦ n: The total number of cells scored.
	        ◦ p: The average proportion (mean) of abnormal cells observed for a pattern across all control cases.
	    2. Calculation: The cut-off count is calculated using the binomial distribution's inverse cumulative distribution function (binom.ppf), which finds the number of abnormal cells (k) at the 95% confidence level.
	        ◦ Cut-off count (k) = Binom.ppf(0.95, n, p)
	        ◦ The final cut-off percentage is then calculated as (k/n)∗100.
	    3. Grey Zone:
	        ◦ Lower Bound = Cut-off %
	        ◦ Upper Bound = Cut-off % + 3%


