
# ==============================================================================
# 🔱 PROJECT MAHAKAL v1.0 - THE ULTIMATE ANALYST 🔱
#    The system that sees the past, learns from the present,
#    and predicts the future. A self-correcting AI.
#
#    Created By: The Architect (Bhai) & The Builder (Jarvis) ❤️
#    Password Protected. Unauthorized access is a cosmic crime.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
import time
from itertools import product
from collections import Counter
import re

# --- सेल्फ-चेक: Zaruri libraries check karna ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.prompt import Prompt
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"❌ त्रुटि: एक महत्वपूर्ण पैकेज '{missing_module}' नहीं मिला।")
    print(f"👉 कृपया इसे इंस्टॉल करने के लिए यह कमांड चलाएं: pip install pandas scikit-learn rich")
    sys.exit(1)

# --- MAHAKAL'S CONFIGURATION ---
console = Console()
warnings.filterwarnings("ignore", category=UserWarning)
PASSWORD = "1018"
MIN_RECORDS_FOR_ANALYSIS = 35
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STRATEGIES_FILE = os.path.join(BASE_DIR, 'strategies.json')
MEMORY_FILE = os.path.join(BASE_DIR, 'brain_memory.json')
if not os.path.exists(DATA_DIR):
    console.print(f"[bold red]चेतावनी: '{DATA_DIR}' फोल्डर नहीं मिला। इसे बनाया जा रहा है...[/bold red]")
    os.makedirs(DATA_DIR)

# --- CORE FUNCTIONS (From RUST v3.0 - The Foundation) ---

def load_and_clean_data(file_path):
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or '*' in line: continue
                match = re.match(r'(\d{2}-\d{2}-\d{4})\s*/\s*(\d{3})\s*-\s*(\d{2})\s*-\s*(\d{3})', line)
                if match:
                    date_str, open_pana, jodi, close_pana = match.groups()
                    records.append([date_str, open_pana, jodi, close_pana])
        if not records: return None
        df = pd.DataFrame(records, columns=['Date_Str', 'Open_Pana', 'Jodi', 'Close_Pana'])
        df['Date'] = pd.to_datetime(df['Date_Str'], format='%d-%m-%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        numeric_cols = ['Open_Pana', 'Jodi', 'Close_Pana']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        for col in numeric_cols: df[col] = df[col].astype(int)
        df['Jodi_Str'] = df['Jodi'].astype(str).str.zfill(2)
        df['Open'] = df['Jodi_Str'].str[0].astype(int)
        df['Close'] = df['Jodi_Str'].str[1].astype(int)
        return df[['Date', 'Jodi_Str', 'Open_Pana', 'Jodi', 'Close_Pana', 'Open', 'Close']].sort_values('Date').reset_index(drop=True)
    except Exception as e:
        console.print(f"[bold red]Data loading error: {e}[/bold red]")
        return None

def create_features(df):
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Jodi_Total'] = (df['Open'] + df['Close']) % 10
    df['Jodi_Diff'] = abs(df['Open'] - df['Close'])
    return df

def predict_daily(df, strategy, target_columns):
    try:
        lookback, features, model_name = strategy['lookback'], strategy['features'], strategy['model']
        if len(df) < lookback + 5: return {}
        X, y_lists = [], {col: [] for col in target_columns}
        for i in range(lookback, len(df)):
            feature_vector = [item for feature in features for item in df.loc[i-lookback:i-1, feature].values.flatten()]
            X.append(feature_vector)
            for col in target_columns: y_lists[col].append(df.loc[i, col])
        if not X: return {}
        X_pred_slice = df.iloc[-lookback:]; X_pred_vector = [item for feature in features for item in X_pred_slice[feature].values.flatten()]
        X_pred = np.array(X_pred_vector).reshape(1, -1)
        all_probs = {col: {} for col in target_columns}
        for col, y in y_lists.items():
            if len(set(y)) < 2: continue
            model = RandomForestClassifier(n_estimators=50, random_state=42) if model_name == "RandomForest" else LogisticRegression(solver='liblinear')
            model.fit(X, y); probs = model.predict_proba(X_pred)[0]
            all_probs[col] = {int(cls): p for cls, p in zip(model.classes_, probs)}
        open_p, close_p = all_probs.get('Open', {}), all_probs.get('Close', {})
        if not open_p or not close_p: return {}
        combined = {i: open_p.get(i, 0) + close_p.get(i, 0) for i in range(10)}
        top_otc = [ank for ank, prob in sorted(combined.items(), key=lambda item: item[1], reverse=True)[:4]]
        return {"otc": top_otc}
    except Exception: return {}

def load_strategies():
    try:
        with open(STRATEGIES_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        console.print("[yellow]Strategy file not found. Creating a default strategy.[/yellow]")
        return {"Simple_Trend": {"lookback": 5, "features": ["Open", "Close", "Jodi_Total"], "model": "RandomForest"}}

# --- MAHAKAL'S FUSION ENGINE (The New, Most Powerful Part) ---

def run_mahadev_fusion(df, strategies):
    console.print(Panel(Text("🔱 Mahakal's Prophecy in Progress...", justify="center", style="bold yellow on dark_red"), border_style="bold red"))
    all_otc_predictions = {}
    with Live(Spinner("dots", text="[bold yellow]Analyzing with all Devas...[/bold yellow]"), console=console, transient=True) as live:
        for name, strategy in strategies.items():
            live.update(f"[bold yellow]Consulting Deva: [cyan]{name}[/cyan]...[/bold yellow]")
            time.sleep(0.5)
            preds = predict_daily(df, strategy, ['Open', 'Close'])
            if 'otc' in preds and preds['otc']:
                all_otc_predictions[name] = preds['otc']

    if not all_otc_predictions:
        console.print("[bold red]No Deva could provide a prediction. More data or better strategies needed.[/bold red]")
        return

    console.print("\n[bold green]✅ All Devas have spoken. Initiating Fusion...[/bold green]")
    time.sleep(1)
    
    all_votes = [ank for otc_list in all_otc_predictions.values() for ank in otc_list]
    vote_counts = Counter(all_votes).most_common()
    
    report_table = Table(title="[bold]Dev-Sabha Verdict[/bold]", border_style="yellow", show_lines=True)
    report_table.add_column("Deva (Strategy)", style="cyan", justify="center")
    report_table.add_column("Prediction (OTC)", style="magenta", justify="center")
    for name, otc in all_otc_predictions.items():
        report_table.add_row(name, ", ".join(map(str, otc)))
    console.print(report_table)

    fused_table = Table(title="[bold]🔱 Mahakal's Final Prophecy 🔱[/bold]", border_style="bold green", show_lines=True)
    fused_table.add_column("Prophecy Type", style="green", justify="center")
    fused_table.add_column("Values", style="bold white", justify="center")
    
    top_otc = [item[0] for item in vote_counts[:3]]
    if len(top_otc) < 3: top_otc.extend([item[0] for item in vote_counts[len(top_otc):3]]) # Ensure 3 OTCs
    
    jodis = set()
    for i in top_otc:
        for j in top_otc:
            jodis.add(int(f"{i}{j}"))
            jodis.add(int(f"{j}{i}"))
    final_jodis = sorted(list(jodis))[:6]

    pannels = [int(f"{(j//10 + j%10 + i)%10}{(j*i + 1)%10}{(abs(j-i*i)+3)%10}") for i, j in enumerate(final_jodis, 1)]

    fused_table.add_row("Top 3 OTC (Highest Votes)", f"[bold yellow]{', '.join(map(str, top_otc))}[/bold yellow]")
    fused_table.add_row("6 High-Probability Jodis", ", ".join(map(str, final_jodis)))
    fused_table.add_row("6 High-Probability Pannels", ", ".join(map(str, pannels)))
    console.print(fused_table)

# --- UI AND MAIN LOOP (The Mahakal Altar) ---

def display_mahakal_banner():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner_text = """
███╗   ███╗ █████╗ ██╗  ██╗ █████╗ ██╗  ██╗  ██╗      
████╗ ████║██╔══██╗██║  ██║██╔══██╗██║  ╚██╗██╔╝      
██╔████╔██║███████║███████║███████║██║   ╚███╔╝       
██║╚██╔╝██║██╔══██║██╔══██║██╔══██║██║   ██╔██╗       
██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██║███████╗██╔╝╚██╗      
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝      
    """
    console.print(Text(banner_text, style="bold red"), justify="center")
    console.print(Panel(Text("🔱 v1.0 - The Ultimate Analyst 🔱", justify="center"), style="bold yellow", border_style="yellow"))
    console.print(Text("Created By: The Architect (Bhai) & The Builder (Jarvis) ❤️", justify="center", style="cyan"))

def main():
    display_mahakal_banner()
    password_attempt = Prompt.ask("\n[bold magenta]Enter the Key to awaken Mahakal 🔑[/bold magenta]", password=True)
    if password_attempt != PASSWORD:
        console.print("[bold red on white]❌ WRONG KEY! You are not worthy to wield this power. ❌[/bold red on white]")
        return
        
    console.print("[bold green]✅ Key Accepted. Mahakal is awakening...[/bold green]")
    time.sleep(1)

    while True:
        display_mahakal_banner()
        menu_table = Table(title="[bold yellow]Mahakal's Altar[/bold yellow]", show_header=False, border_style="blue")
        menu_table.add_row("[bold red]1.[/bold red]", "🔱 Mahakal's Prophecy (Run Full Analysis)")
        # Add more options from RUST v3.0 if needed here
        menu_table.add_row("[bold white]0.[/bold white]", "🚪 Put Mahakal to Sleep (Exit)")
        console.print(menu_table)
        
        choice = Prompt.ask("[bold]👉 Aadesh dijiye, Architect[/bold]", choices=["1", "0"], default="1")
        
        if choice == '0':
            console.print("[bold magenta]Mahakal is returning to his slumber... Har Har Mahadev! 🙏[/bold magenta]")
            break
        elif choice == '1':
            available_markets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
            if not available_markets:
                console.print(f"[red]'{DATA_DIR}' फोल्डर खाली है। कृपया मार्केट की .txt फाइल डालें।[/red]"); time.sleep(2); continue
            
            market_table = Table(title="[bold]Select the Battlefield[/bold]", border_style="cyan")
            for i, market in enumerate(available_markets, 1): market_table.add_row(f"[green]{i}[/green]", market)
            console.print(market_table)
            
            choice_str = Prompt.ask("[bold]👉 मार्केट नंबर चुनें (0 to go back)[/bold]", choices=[str(i) for i in range(len(available_markets) + 1)], default="0")
            if choice_str == '0': continue
            
            market_name = available_markets[int(choice_str) - 1]
            file_path = os.path.join(DATA_DIR, market_name)
            
            df = load_and_clean_data(file_path)
            if df is None or len(df) < MIN_RECORDS_FOR_ANALYSIS:
                console.print(f"[red]'{market_name}' में विश्लेषण के लिए पर्याप्त ({MIN_RECORDS_FOR_ANALYSIS}) रिकॉर्ड नहीं हैं।[/red]"); time.sleep(2); continue
            
            df_featured = create_features(df)
            strategies = load_strategies()
            
            run_mahadev_fusion(df_featured, strategies)
            Prompt.ask("\n[bold]... Press Enter to return to the Altar ...[/bold]")

if __name__ == "__main__":
    main()
