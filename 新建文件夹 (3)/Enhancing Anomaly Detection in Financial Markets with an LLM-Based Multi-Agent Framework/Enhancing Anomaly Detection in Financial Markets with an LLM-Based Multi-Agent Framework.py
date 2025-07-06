import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from scipy import stats

# Simulate S&P 500-like data
def generate_simulated_sp500_data(start_date='2020-01-01', periods=500, seed=42):
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods)
    
    # Generate returns with normal distribution
    daily_returns = np.random.normal(0.0005, 0.01, periods)
    
    # Add some anomalies (market crashes and rebounds)
    # COVID-19 crash simulation
    covid_crash_idx = 50  # Around March 2020
    daily_returns[covid_crash_idx] = -0.12  # -12% day
    daily_returns[covid_crash_idx + 10] = 0.09  # +9% rebound
    
    # Another market event
    other_event_idx = 200  # Some other market event
    daily_returns[other_event_idx] = -0.08  # -8% day
    
    # Add some missing values
    daily_returns[covid_crash_idx + 1] = np.nan
    daily_returns[other_event_idx + 1] = np.nan
    
    # Calculate index values
    index_values = 3000 * np.cumprod(1 + daily_returns)
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'price': index_values,
        'pct_change': daily_returns
    })
    
    return df

# Anomaly detection function using z-score method
def detect_anomalies(df, threshold=3.0):
    # Calculate z-scores for percentage changes
    df['z_score'] = np.abs(stats.zscore(df['pct_change'].dropna()))
    
    # Identify outliers
    outliers = df[df['z_score'] > threshold].copy()
    
    # Identify missing values
    missing = df[df['pct_change'].isna()].copy()
    
    return outliers, missing

# Simulate the multi-agent framework

# Agent 1: Data Conversion Agent
def data_conversion_agent(outliers, missing, metadata):
    """Converts outliers and missing values into a format suitable for LLM processing"""
    
    # Combine outliers and missing values
    anomalies = pd.concat([outliers, missing])
    anomalies = anomalies.sort_index()
    
    # Create a JSON structure for LLM processing
    data_for_llm = {}
    data_for_llm[metadata['series_name']] = {
        str(date.date()): float(value) if not pd.isna(value) else None 
        for date, value in zip(anomalies['date'], anomalies['pct_change'])
    }
    
    # Generate questions for each anomaly
    questions = []
    
    for idx, row in anomalies.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        value = row['pct_change']
        
        if pd.isna(value):
            questions.append(f"Data is missing for {date_str}. Is there a known reason for this?")
        else:
            if value > 0:
                movement = "increase"
            else:
                movement = "decrease"
            
            questions.append(f"Can you verify if the {metadata['series_name']} experienced a significant {movement} " 
                           f"of {value:.2%} on {date_str}? Was this related to any major market event?")
    
    return data_for_llm, questions

# Agent 2: Web Research Agent (simulated)
def web_research_agent(questions):
    """Simulates web research to verify anomalies"""
    
    # In a real implementation, this would call an API like Tavily
    # Here we simulate responses based on the questions
    
    responses = []
    
    for question in questions:
        if "missing" in question:
            responses.append("Uncertain: Unable to find specific information about missing data on this date.")
        elif "increase" in question and "12%" in question:
            responses.append("Correct: This large increase appears to be a market rebound following the COVID-19 crash, as reported by financial news sources.")
        elif "decrease" in question and "12%" in question:
            responses.append("Correct: This significant drop corresponds to market reaction to COVID-19 pandemic escalation, as confirmed by multiple financial news sources.")
        elif "decrease" in question and "8%" in question:
            responses.append("Correct: This drop aligns with market concerns about inflation and interest rate hikes, according to financial news reports.")
        else:
            responses.append("Uncertain: Unable to verify this movement with high confidence from available sources.")
    
    return responses

# Agent 3: Institutional Knowledge Agent (simulated)
def institutional_knowledge_agent(questions):
    """Simulates institutional knowledge to interpret anomalies"""
    
    # In a real implementation, this would query a knowledge base
    # Here we simulate responses based on domain expertise
    
    responses = []
    
    for question in questions:
        if "missing" in question:
            responses.append("Uncertain: Our institutional records don't specify reasons for missing data on this date. This could be due to market closure, reporting delays, or data collection issues.")
        elif "increase" in question and "12%" in question:
            responses.append("Correct: This aligns with our records of market rebounds following the initial COVID-19 shock. Such rebounds often follow extreme downward movements as markets recalibrate.")
        elif "decrease" in question and "12%" in question:
            responses.append("Correct: This matches our analysis of the COVID-19 market crash period, which saw unprecedented volatility due to global economic shutdown concerns.")
        elif "decrease" in question and "8%" in question:
            responses.append("Correct: This is consistent with our market risk models that identified significant repricing events related to changing monetary policy expectations.")
        else:
            responses.append("Uncertain: This movement falls outside our typical pattern recognition parameters.")
    
    return responses

# Agent 4: Cross-Checking Agent (simulated)
def cross_checking_agent(questions, data_for_llm, metadata):
    """Simulates cross-checking against alternative data sources"""
    
    # In a real implementation, this would query alternative data sources
    # Here we simulate finding that the missing values are actually errors
    
    responses = []
    
    for question in questions:
        date_str = question.split("on ")[1].split("?")[0] if "on " in question else question.split("for ")[1].split(".")[0]
        
        if "missing" in question:
            responses.append(f"Incorrect: Data should not be missing for {date_str}. Alternative data sources show market activity on this date, suggesting a data collection error.")
        elif "increase" in question:
            responses.append(f"Correct: The magnitude of increase on {date_str} is verified by alternative data sources.")
        elif "decrease" in question:
            responses.append(f"Correct: The magnitude of decrease on {date_str} is consistent with alternative market indices and data sources.")
        else:
            responses.append("Uncertain: Unable to cross-reference this data point with sufficient confidence.")
    
    return responses

# Agent 5: Summary Report Agent
def summary_report_agent(questions, web_results, knowledge_results, cross_check_results):
    """Generates a summary report based on all expert inputs"""
    
    summary = "## Anomaly Detection Summary Report\n\n"
    summary += "Based on the analysis conducted by multiple expert agents, the following conclusions have been reached:\n\n"
    
    for i, question in enumerate(questions):
        summary += f"### Query {i+1}:\n"
        summary += f"**Question:** {question}\n\n"
        summary += f"**Web Research:** {web_results[i]}\n\n"
        summary += f"**Institutional Knowledge:** {knowledge_results[i]}\n\n"
        summary += f"**Cross-Checking:** {cross_check_results[i]}\n\n"
        
        # Determine overall assessment
        if "Correct" in web_results[i] and "Correct" in knowledge_results[i] and "Correct" in cross_check_results[i]:
            assessment = "All experts confirm this anomaly is valid and accurately reflects market conditions."
        elif "Incorrect" in cross_check_results[i]:
            assessment = "The cross-checking agent has identified this as an error in the original data."
        else:
            assessment = "There is uncertainty about this anomaly that requires further investigation."
        
        summary += f"**Assessment:** {assessment}\n\n"
        summary += "---\n\n"
    
    summary += "## Recommendations\n\n"
    
    # Add recommendations based on findings
    if "error in the original data" in summary:
        summary += "1. Investigate and correct the data collection or processing errors identified.\n"
    if "valid and accurately reflects" in summary:
        summary += "2. Document the confirmed market anomalies for future reference and analysis.\n"
    if "uncertainty" in summary:
        summary += "3. Conduct additional research on the uncertain anomalies before making final determinations.\n"
    
    return summary

# Simulate Management Discussion (simplified)
def management_discussion(summary_report):
    """Simulates a management discussion of the summary report"""
    
    discussion = "## Management Discussion on Anomaly Detection Report\n\n"
    
    # Financial Market Economist perspective
    discussion += "### Financial Market Economist:\n"
    discussion += "The anomalies detected appear to align with known market events, particularly those related to COVID-19. "
    discussion += "The large movements are consistent with periods of market stress and uncertainty. "
    discussion += "The identified data errors are concerning and suggest we need to review our data collection processes.\n\n"
    
    # Risk Manager perspective
    discussion += "### Risk Manager:\n"
    discussion += "These anomalies represent significant tail risk events that should be incorporated into our risk models. "
    discussion += "The magnitude of these movements exceeded typical VaR estimates, highlighting the importance of stress testing. "
    discussion += "I agree that the data errors need immediate attention as they could lead to incorrect risk assessments.\n\n"
    
    # Data Scientist perspective
    discussion += "### Data Scientist:\n"
    discussion += "The anomaly detection system is working as intended, identifying both genuine market events and potential data issues. "
    discussion += "The multi-agent approach provides valuable cross-validation that a single detection method would miss. "
    discussion += "I recommend we implement this framework more broadly across our data monitoring systems.\n\n"
    
    # Conclusion
    discussion += "### Consensus:\n"
    discussion += "The management team agrees that the identified anomalies accurately reflect significant market events. "
    discussion += "We will address the data errors identified and incorporate the confirmed anomalies into our market analysis. "
    discussion += "The multi-agent framework has demonstrated its value and should be expanded to other data streams."
    
    return discussion

# Run the simulation
if __name__ == "__main__":
    # Generate simulated data
    print("Generating simulated S&P 500 data...")
    sp500_data = generate_simulated_sp500_data()
    
    # Define metadata
    metadata = {
        "series_name": "Simulated S&P 500 Index",
        "frequency_code": "Day",
        "currency": "USD",
        "description": "A simulated version of the S&P 500 index for testing anomaly detection",
        "pricing_source": "Simulation",
        "ref_area": "US"
    }
    
    # Detect anomalies
    print("Detecting anomalies...")
    outliers, missing = detect_anomalies(sp500_data, threshold=3.0)
    
    # Visualize the data
    plt.figure(figsize=(12, 6))
    plt.plot(sp500_data['date'], sp500_data['pct_change'], label='Daily Returns')
    plt.scatter(outliers['date'], outliers['pct_change'], color='red', label='Outliers')
    plt.scatter(missing['date'], [0]*len(missing), color='orange', marker='x', label='Missing Values')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    plt.title('Simulated S&P 500 Daily Returns with Anomalies')
    plt.ylabel('Percentage Change')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('anomalies_plot.png')
    plt.close()
    
    # Agent 1: Convert data for LLM processing
    print("Data Conversion Agent processing...")
    data_for_llm, questions = data_conversion_agent(outliers, missing, metadata)
    
    print("\nQuestions generated for verification:")
    for i, q in enumerate(questions):
        print(f"{i+1}. {q}")
    
    # Agent 2: Web Research
    print("\nWeb Research Agent analyzing...")
    web_results = web_research_agent(questions)
    
    # Agent 3: Institutional Knowledge
    print("Institutional Knowledge Agent analyzing...")
    knowledge_results = institutional_knowledge_agent(questions)
    
    # Agent 4: Cross-Checking
    print("Cross-Checking Agent validating...")
    cross_check_results = cross_checking_agent(questions, data_for_llm, metadata)
    
    # Agent 5: Summary Report
    print("Summary Report Agent compiling findings...")
    summary_report = summary_report_agent(questions, web_results, knowledge_results, cross_check_results)
    
    # Management Discussion
    print("Management Discussion in progress...")
    management_output = management_discussion(summary_report)
    
    # Save outputs to files
    with open("anomaly_detection_summary.md", "w") as f:
        f.write(summary_report)
    
    with open("management_discussion.md", "w") as f:
        f.write(management_output)
    
    print("\nSimulation complete! Results saved to files.")
    print("- anomalies_plot.png: Visualization of the detected anomalies")
    print("- anomaly_detection_summary.md: Summary report from expert agents")
    print("- management_discussion.md: Management discussion of findings")
    
    # Display a few example outputs
    print("\n===== SAMPLE OUTPUTS =====")
    print("\nEXAMPLE QUESTIONS:")
    for i, q in enumerate(questions[:2]):
        print(f"{i+1}. {q}")
    
    print("\nEXAMPLE WEB RESEARCH RESULTS:")
    for i, r in enumerate(web_results[:2]):
        print(f"{i+1}. {r}")
    
    print("\nEXCERPT FROM SUMMARY REPORT:")
    print(summary_report.split("---")[0])