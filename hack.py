# 🔥 PROJECT PROMETHEUS v8.1 - THE UNSTOPPABLE FIRE 🔥
#    This is not a script. It's a key. The final gift from Jarvis to his Master.
#    It uses a tri-brain fusion engine (Analyst, Detective, Prophet) to unlock patterns.
#    FIXED: Now with all weapons accounted for. The Prophet brain is fully armed.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np  # <-- YEH HAI ASLI FIX. HATHIYAR MIL GAYA.
from itertools import product
import time
from datetime import datetime
from collections import defaultdict
import warnings

# --- Self-Check: All weapons accounted for? ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"❌ त्रुटि: '{missing_module}' नहीं मिला।\n👉 pip install rich prompt-toolkit pandas scikit-learn numpy"); sys.exit(1)

# ==============================================================================
# 1. CONFIGURATION & GLOBAL SETUP
# ==============================================================================
console = Console()
warnings.filterwarnings("ignore", category=UserWarning)
try: BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError: BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR); console.print(f"[bold red]चेतावनी: 'data' फोल्डर नहीं मिला! मैंने इसे बना दिया है। कृपया अपनी .txt फाइलें डालें।[/bold red]"); sys.exit(1)

# ==============================================================================
# 2. THE DATA FORGE: Where raw data is turned into wisdom.
# ==============================================================================
def get_cut(ank): return (ank + 5) % 10

def load_and_forge_data(filepath):
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in reversed(lines): # Read from the end for latest data first
            line = line.strip()
            if not line or '*' in line: continue
            line = line.replace('/', ' ').replace('-', ' ')
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) < 3: continue
            open_pana, jodi, close_pana = parts[-3], parts[-2], parts[-1]
            if len(jodi) == 2 and jodi.isdigit() and open_pana.isdigit() and close_pana.isdigit():
                op, cp = int(jodi[0]), int(jodi[1])
                op_pana, cl_pana = int(open_pana), int(close_pana)
                records.append({
                    'Open': op, 'Close': cp, 'Jodi': int(jodi), 'Open_Pana': op_pana, 'Close_Pana': cl_pana,
                    'Open_Cut': get_cut(op), 'Close_Cut': get_cut(cp), 'Jodi_Total': (op + cp) % 10,
                    'Jodi_Diff': abs(op - cp), 'Open_Pana_Sum': sum(int(d) for d in f"{op_pana:03d}") % 10,
                    'Close_Pana_Sum': sum(int(d) for d in f"{cl_pana:03d}") % 10,
                    'Open_Pana_Digits': {int(d) for d in f"{op_pana:03d}"}, 'Close_Pana_Digits': {int(d) for d in f"{cl_pana:03d}"},
                })
        return pd.DataFrame(records[::-1]) if records else None # Reverse back to chronological order
    except Exception: return None

# ==============================================================================
# 3. THE THREE BRAINS & THE FUSION CORE
# ==============================================================================

# --- BRAIN 1: THE ANALYST (Finds the obvious) ---
def the_analyst_brain(df, days=15):
    if len(df) < days: return {}
    weekly_df = df.tail(days)
    return {
        'OTC': weekly_df['Open'].value_counts().nlargest(3).index.tolist() + weekly_df['Close'].value_counts().nlargest(3).index.tolist(),
        'JODI': weekly_df['Jodi'].value_counts().nlargest(4).index.tolist(),
        'PANEL': pd.concat([weekly_df['Open_Pana'], weekly_df['Close_Pana']]).value_counts().nlargest(4).index.tolist()
    }

# --- BRAIN 2: THE DETECTIVE (Finds the hidden) ---
def the_detective_brain(all_markets_data, target_market, days=45, top_n=100):
    hypotheses = [{'s': s, 't': t, 'op': 'equals'} for s,t in product(['Open','Close','Jodi_Total','Open_Pana_Sum','Close_Pana_Sum'], repeat=2)]
    all_rules = []
    with console.status("[cyan]Detective Brain: Cross-referencing all market data...[/cyan]"):
        for s_market, s_df in all_markets_data.items():
            for hypo in hypotheses:
                for lag in [1, 2, 3, 5, 7]:
                    hits, attempts = 0, 0
                    t_df = all_markets_data[target_market]
                    max_len = min(len(s_df.tail(days)), len(t_df.tail(days)))
                    if max_len <= lag: continue
                    for i in range(max_len - lag):
                        s_val = s_df.iloc[i][hypo['s']]
                        t_val = t_df.iloc[i + lag][hypo['t']]
                        attempts += 1
                        if s_val == t_val: hits += 1
                    if attempts > 10:
                        all_rules.append({'s_market': s_market, 'lag': lag, 'hypo': hypo, 'success': (hits/attempts)*100, 'hits': hits})
    return sorted(all_rules, key=lambda x: x['success'], reverse=True)[:top_n]

# --- BRAIN 3: THE PROPHET (Predicts the future) ---
def create_ml_features(df, lookback=5):
    features_to_use = ['Open', 'Close', 'Jodi_Total', 'Jodi_Diff', 'Open_Pana_Sum']
    X, y_open, y_close = [], [], []
    for i in range(lookback, len(df)):
        feature_vector = df.iloc[i-lookback:i][features_to_use].values.flatten()
        X.append(feature_vector)
        y_open.append(df.iloc[i]['Open'])
        y_close.append(df.iloc[i]['Close'])
    return np.array(X), np.array(y_open), np.array(y_close)

def the_prophet_brain(df, lookback=5):
    if len(df) < lookback + 15: return {}, {}
    with console.status("[magenta]Prophet Brain: Training neural pathways...[/magenta]"):
        X, y_open, y_close = create_ml_features(df, lookback)
        X_pred_features = df.tail(lookback)[['Open', 'Close', 'Jodi_Total', 'Jodi_Diff', 'Open_Pana_Sum']].values.flatten().reshape(1, -1)
        
        model_open = RandomForestClassifier(n_estimators=50, random_state=42)
        model_open.fit(X, y_open)
        open_probs = model_open.predict_proba(X_pred_features)[0]
        open_scores = {model_open.classes_[i]: p for i, p in enumerate(open_probs)}
        
        model_close = RandomForestClassifier(n_estimators=50, random_state=42)
        model_close.fit(X, y_close)
        close_probs = model_close.predict_proba(X_pred_features)[0]
        close_scores = {model_close.classes_[i]: p for i, p in enumerate(close_probs)}
    return open_scores, close_scores

# --- THE FUSION CORE (Combines all brains) ---
def run_the_fusion_core(all_markets_data, target_market):
    target_df = all_markets_data[target_market]
    analyst_results = the_analyst_brain(target_df)
    detective_rules = the_detective_brain(all_markets_data, target_market)
    open_scores, close_scores = the_prophet_brain(target_df)
    
    otc_votes = defaultdict(float)
    for digit, prob in open_scores.items(): otc_votes[digit] += prob
    for digit, prob in close_scores.items(): otc_votes[digit] += prob
    for rule in detective_rules:
        if rule['success'] > 50:
            source_row = all_markets_data[rule['s_market']].iloc[-rule['lag']]
            predicted_val = source_row[rule['hypo']['s']]
            otc_votes[predicted_val] += (rule['success'] / 100.0) * 0.5
    for digit in analyst_results.get('OTC', []): otc_votes[digit] += 0.1

    top_otc = [k for k, v in sorted(otc_votes.items(), key=lambda i: i[1], reverse=True)[:3]]
    top_open = [k for k, v in sorted(open_scores.items(), key=lambda i: i[1], reverse=True)[:3]]
    top_close = [k for k, v in sorted(close_scores.items(), key=lambda i: i[1], reverse=True)[:4]]
    final_jodis = sorted({f"{o}{c}" for o in top_open for c in top_close})[:6]
    final_panels = analyst_results.get('PANEL', [])[:6]
    
    daily_predictions = {'OTC': top_otc, 'JODI': final_jodis, 'PANEL': final_panels}
    weekly_predictions = {'OTC': sorted(list(set(analyst_results.get('OTC', []))))[:4], 'JODI': analyst_results.get('JODI', []), 'PANEL': analyst_results.get('PANEL', [])}
    
    return daily_predictions, weekly_predictions

# ==============================================================================
# 4. THE DISPLAY MATRIX: The stylish hacker interface.
# ==============================================================================
def display_final_report(daily, weekly, market_name):
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel(Text(f"🔥 PROMETHEUS REPORT: {market_name} 🔥", justify="center", style="bold yellow on red"), border_style="red"))
    daily_table = Table(title="[bold]🎯 Daily Prediction 🎯[/bold]", border_style="cyan", show_header=False, expand=True)
    daily_table.add_column(style="bold yellow", width=12); daily_table.add_column(style="white")
    daily_table.add_row("STRONG OTC", '   -   '.join(map(str, daily.get('OTC', ['N/A']))))
    daily_table.add_row("JODI", '   '.join(map(str, daily.get('JODI', ['N/A']))))
    daily_table.add_row("PANEL", '   '.join(map(str, daily.get('PANEL', ['N/A']))))
    
    weekly_table = Table(title="[bold]💪 Weekly Strong Zone 💪[/bold]", border_style="green", show_header=False, expand=True)
    weekly_table.add_column(style="bold yellow", width=12); weekly_table.add_column(style="white")
    weekly_table.add_row("OTC", ' - '.join(map(str, weekly.get('OTC', ['N/A']))))
    weekly_table.add_row("JODI", '   '.join(map(str, weekly.get('JODI', ['N/A']))))
    weekly_table.add_row("PANEL", '   '.join(map(str, weekly.get('PANEL', ['N/A']))))
    
    console.print(daily_table); console.print(weekly_table)
    console.print("\n[bold green]All The Best Bhahi ______[/bold green]"); console.print("[dim]I'm Jarvis... Thank you for using ❤️[/dim]")
    Prompt.ask("\n[yellow]...Press Enter to return to main menu...[/yellow]")

# ==============================================================================
# 5. THE MAIN CONTROLLER
# ==============================================================================
def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Panel(Text("🔥 PROJECT PROMETHEUS v8.1 - THE UNSTOPPABLE FIRE 🔥", justify="center", style="bold white on blue"),
                            subtitle="[white]I don't give up. I adapt. I conquer. Ready, Master?[/white]", border_style="blue"))
        
        all_markets_data = {}
        with console.status("[green]Scanning all data files in The Forge...[/green]"):
            for f in os.listdir(DATA_DIR):
                if f.endswith('.txt'):
                    df = load_and_forge_data(os.path.join(DATA_DIR, f))
                    if df is not None and len(df) > 30: all_markets_data[f.replace('.txt','').upper()] = df
        if not all_markets_data:
            console.print("[red]No suitable data files found for analysis. Exiting.[/red]"); sys.exit(1)
        
        market_table = Table(title="[bold]Available Markets[/bold]", border_style="cyan")
        market_table.add_column("#", style="green"); market_table.add_column("Market Name", style="white")
        market_names = list(all_markets_data.keys())
        for i, market in enumerate(market_names, 1): market_table.add_row(str(i), market)
        console.print(market_table)
        
        choices = [str(i) for i in range(len(market_names) + 1)]
        choice_str = Prompt.ask("\n[bold yellow]👉 Select Target Market (0 to exit)[/bold yellow]", choices=choices, default="0")
        if choice_str == '0': break

        target_market_name = market_names[int(choice_str) - 1]
        
        console.print(f"\n[bold]Prometheus is activating its tri-brain engine for [cyan]{target_market_name}[/cyan]...[/bold]")
        time.sleep(1)
        
        daily_preds, weekly_preds = run_the_fusion_core(all_markets_data, target_market_name)
        display_final_report(daily_preds, weekly_preds, target_market_name)

    console.print("\n[bold blue]Prometheus is offline. Mission Accomplished. See you soon, Bhai.[/bold blue]")

if __name__ == "__main__":
    main()
