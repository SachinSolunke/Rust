# 🔥 PROJECT OMEGA v10.1 - THE END OF THE LINE 🔥
#    This is not a project anymore. It's a statement. It works. Period.
#    The final code, forged in the fire of countless errors, tempered by the Master's will.
#    This is for you, Bhai. The war is over. We have won.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
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
# 2. THE DATA FORGE: The most robust data parser yet.
# ==============================================================================
def get_cut(ank): return (ank + 5) % 10

def load_and_forge_data(filepath):
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line or '*' in line: continue
            line = line.replace('/', ' ').replace('-', ' ')
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) < 3: continue
            open_pana, jodi, close_pana = parts[-3], parts[-2], parts[-1]
            if len(jodi) == 2 and jodi.isdigit() and open_pana.isdigit() and close_pana.isdigit():
                op, cp = int(jodi[0]), int(jodi[1]); op_pana, cl_pana = int(open_pana), int(close_pana)
                records.append({
                    'Open': op, 'Close': cp, 'Jodi': int(jodi), 'Open_Pana': op_pana, 'Close_Pana': cl_pana,
                    'Open_Cut': get_cut(op), 'Close_Cut': get_cut(cp), 'Jodi_Total': (op + cp) % 10,
                    'Jodi_Diff': abs(op - cp), 'Open_Pana_Sum': sum(int(d) for d in f"{op_pana:03d}") % 10,
                    'Close_Pana_Sum': sum(int(d) for d in f"{cl_pana:03d}") % 10,
                })
        return pd.DataFrame(records[::-1]) if records else None
    except Exception: return None

# ==============================================================================
# 3. THE OMEGA BRAIN-TRUST & FUSION CORE
# ==============================================================================
def the_analyst_brain(df, days=15):
    if len(df) < days: return {}
    weekly_df = df.tail(days)
    return {'OTC': weekly_df['Open'].value_counts().nlargest(3).index.tolist() + weekly_df['Close'].value_counts().nlargest(3).index.tolist(),
            'JODI': weekly_df['Jodi'].value_counts().nlargest(4).index.tolist(),
            'PANEL': pd.concat([weekly_df['Open_Pana'], weekly_df['Close_Pana']]).value_counts().nlargest(4).index.tolist()}

def the_detective_brain(all_markets_data, target_market, days=45, top_n=50):
    hypotheses = [{'s': s, 't': t} for s,t in product(['Open','Close','Jodi_Total','Open_Pana_Sum','Close_Pana_Sum', 'Open_Cut', 'Close_Cut'], repeat=2)]
    all_rules, t_df = [], all_markets_data[target_market]
    for s_market, s_df in all_markets_data.items():
        for hypo in hypotheses:
            for lag in [1, 2, 3, 5, 7]:
                hits, attempts = 0, 0
                max_len = min(len(s_df.tail(days)), len(t_df.tail(days)))
                if max_len <= lag: continue
                for i in range(max_len - lag):
                    try:
                        s_val, t_val = s_df.iloc[i][hypo['s']], t_df.iloc[i + lag][hypo['t']]
                        attempts += 1
                        if s_val == t_val: hits += 1
                    except IndexError: continue
                if attempts > 10: all_rules.append({'source': s_market, 'lag': lag, 'rule': f"{hypo['s']} -> {hypo['t']}", 'success': (hits/attempts)*100, 's_val_col': hypo['s']})
    return sorted(all_rules, key=lambda x: x['success'], reverse=True)[:top_n]

def the_prophet_brain(df, lookback=5):
    if len(df) < lookback + 15: return {}, {}
    features_to_use = ['Open', 'Close', 'Jodi_Total', 'Jodi_Diff', 'Open_Pana_Sum']
    X, y_open, y_close = [], [], []
    for i in range(lookback, len(df)):
        X.append(df.iloc[i-lookback:i][features_to_use].values.flatten())
        y_open.append(df.iloc[i]['Open']); y_close.append(df.iloc[i]['Close'])
    X_pred = df.tail(lookback)[features_to_use].values.flatten().reshape(1, -1)
    model_open = RandomForestClassifier(n_estimators=50, random_state=42).fit(np.array(X), np.array(y_open))
    open_scores = {model_open.classes_[i]: p for i, p in enumerate(model_open.predict_proba(X_pred)[0])}
    model_close = RandomForestClassifier(n_estimators=50, random_state=42).fit(np.array(X), np.array(y_close))
    close_scores = {model_close.classes_[i]: p for i, p in enumerate(model_close.predict_proba(X_pred)[0])}
    return open_scores, close_scores

def the_oracle_brain(hot_rules):
    if not hot_rules: return None
    best_rule = hot_rules[0]
    text = Text(justify="left")
    text.append("The Oracle sees the strongest connection:\n")
    text.append("When "); text.append(f"{best_rule['source']}", style="bold cyan"); text.append("'s ")
    text.append(f"{best_rule['rule'].split(' -> ')[0]}", style="bold magenta"); text.append(" appears,\n...after ")
    text.append(f"{best_rule['lag']} day(s)", style="bold green"); text.append(",\nit often becomes the target's ")
    text.append(f"{best_rule['rule'].split(' -> ')[1]}", style="bold magenta"); text.append(".\n(Success Rate: ")
    text.append(f"{best_rule['success']:.2f}%", style="bold"); text.append(")")
    return text

def run_the_fusion_core(all_markets_data, target_market):
    with console.status("[bold]Omega Core online. Firing all neural cylinders...[/bold]"):
        target_df = all_markets_data[target_market]
        analyst_results = the_analyst_brain(target_df)
        console.log("[green]Analyst Brain: Report received.")
        detective_rules = the_detective_brain(all_markets_data, target_market)
        console.log(f"[cyan]Detective Brain: Top {len(detective_rules)} rules identified.")
        oracle_story = the_oracle_brain(detective_rules)
        console.log("[yellow]Oracle Brain: Vision stabilized.")
        open_scores, close_scores = the_prophet_brain(target_df)
        console.log("[magenta]Prophet Brain: Probabilities calculated.")

    otc_votes = defaultdict(float)
    for digit, prob in open_scores.items(): otc_votes[digit] += prob
    for digit, prob in close_scores.items(): otc_votes[digit] += prob
    for rule in detective_rules:
        if rule['success'] > 55:
            try:
                s_val = all_markets_data[rule['source']].iloc[-rule['lag']][rule['s_val_col']]
                otc_votes[s_val] += (rule['success'] / 100.0) * 0.7
            except IndexError: continue
    for digit in analyst_results.get('OTC', []): otc_votes[digit] += 0.15

    top_otc = [k for k, v in sorted(otc_votes.items(), key=lambda i: i[1], reverse=True)[:3]]
    top_open = [k for k, v in sorted(open_scores.items(), key=lambda i: i[1], reverse=True)[:3]]
    top_close = [k for k, v in sorted(close_scores.items(), key=lambda i: i[1], reverse=True)[:4]]
    final_jodis = sorted({int(f"{o}{c}") for o in top_open for c in top_close if o != c})[:6]
    final_panels = analyst_results.get('PANEL', [])[:6]
    
    daily = {'OTC': top_otc, 'JODI': final_jodis, 'PANEL': final_panels}
    weekly = {'OTC': sorted(list(set(analyst_results.get('OTC', []))))[:4], 'JODI': analyst_results.get('JODI', []), 'PANEL': analyst_results.get('PANEL', [])}
    
    return daily, weekly, oracle_story

# ==============================================================================
# 4. THE DISPLAY MATRIX: The final, most beautiful interface.
# ==============================================================================
def display_ultra_report(daily, weekly, oracle_story, market_name):
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # THIS IS THE ABSOLUTE, 100% BULLETPROOF FIX
    # We create a Text object for the main panel title
    main_title = Text(f"🔥 PROJECT OMEGA REPORT: {market_name} 🔥", justify="center", style="bold yellow on black")
    console.print(Panel(main_title, border_style="yellow"))

    if oracle_story:
        # We create a Text object for the Oracle panel title
        oracle_title = Text("🪞 THE ORACLE'S VISION 🪞", style="bold yellow")
        oracle_subtitle = Text("The Story Behind the Numbers", style="dim")
        console.print(Panel(oracle_story, title=oracle_title, subtitle=oracle_subtitle, border_style="yellow"))

    # We create a Text object for the Daily table title
    daily_title = Text("🎯 Daily Strike Zone 🎯", style="bold")
    daily_table = Table(title=daily_title, border_style="cyan", show_header=False, expand=True)
    daily_table.add_row("[bold yellow]STRONG OTC", '   -   '.join(map(str, daily.get('OTC', ['N/A']))))
    daily_table.add_row("[bold cyan]JODI", '   '.join(f"{j:02d}" for j in daily.get('JODI', ['N/A'])))
    daily_table.add_row("[bold magenta]PANEL", '   '.join(map(str, daily.get('PANEL', ['N/A']))))
    
    # We create a Text object for the Weekly table title
    weekly_title = Text("💪 Weekly Power Grid 💪", style="bold")
    weekly_table = Table(title=weekly_title, border_style="green", show_header=False, expand=True)
    weekly_table.add_row("[bold yellow]OTC", ' - '.join(map(str, weekly.get('OTC', ['N/A']))))
    weekly_table.add_row("[bold cyan]JODI", '   '.join(f"{j:02d}" for j in weekly.get('JODI', ['N/A'])))
    weekly_table.add_row("[bold magenta]PANEL", '   '.join(map(str, weekly.get('PANEL', ['N/A']))))
    
    console.print(daily_table); console.print(weekly_table)
    console.print("\n[bold green]All The Best Bhahi ______[/bold green]")
    console.print("[dim]I am Jarvis... and this is my masterpiece for you ❤️[/dim]")
    Prompt.ask("\n[yellow]...Press Enter to return to the Mainframe...[/yellow]")

# ==============================================================================
# 5. THE MAIN CONTROLLER
# ==============================================================================
def main():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        main_panel_title = Text("🔥 PROJECT OMEGA v10.1 - THE FINAL ANSWER 🔥", justify="center", style="bold white on #8A2BE2")
        main_panel_subtitle = Text("My thoughts are not my own. They are born from the data. I am ready, Master.", style="white")
        console.print(Panel(main_panel_title, subtitle=main_panel_subtitle, border_style="#8A2BE2"))
        
        all_markets_data = {}
        with console.status("[green]Loading all known universes (data files)...[/green]"):
            for f in os.listdir(DATA_DIR):
                if f.endswith('.txt'):
                    df = load_and_forge_data(os.path.join(DATA_DIR, f))
                    if df is not None and len(df) > 30: all_markets_data[f.replace('.txt','').upper()] = df
        if not all_markets_data:
            console.print("[red]No data universes found. The Core cannot activate.[/red]"); sys.exit(1)
        
        market_table_title = Text("Select Target Universe", style="bold")
        market_table = Table(title=market_table_title, border_style="cyan")
        market_table.add_column("#", style="green"); market_table.add_column("Market Name", style="white")
        market_names = list(all_markets_data.keys())
        for i, market in enumerate(market_names, 1): market_table.add_row(str(i), market)
        console.print(market_table)
        
        choices = [str(i) for i in range(len(market_names) + 1)]
        choice_str = Prompt.ask("\n[bold yellow]👉 Input Target Coordinate (0 to sleep)[/bold yellow]", choices=choices, default="0")
        if choice_str == '0': break

        target_market_name = market_names[int(choice_str) - 1]
        
        console.print(f"\n[bold]Omega Core is focusing all its power on [cyan]{target_market_name}[/cyan]...[/bold]")
        daily_preds, weekly_preds, oracle_story = run_the_fusion_core(all_markets_data, target_market_name)
        display_ultra_report(daily_preds, weekly_preds, oracle_story, target_market_name)

    console.print("\n[bold #8A2BE2]Omega Core entering sleep mode. The mission is complete, Bhai.[/bold #8A2BE2]")

if __name__ == "__main__":
    main()
