# 🔥 PROJECT SYNAPSE v7.0 - THE LIVING BRAIN 🔥
#    It doesn't fail. It adapts. It learns.
#    This is the final evolution. A brain that never gives up.
# ==============================================================================

import os
import sys
import pandas as pd
from itertools import product
import time
from datetime import datetime, timedelta
from collections import defaultdict

# --- सेल्फ-चेक ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
except ImportError as e:
    missing_module = str(e).split("'")[1]; print(f"❌ त्रुटि: '{missing_module}' नहीं मिला।\n👉 pip install rich prompt-toolkit pandas"); sys.exit(1)

# --- कॉन्फ़िगरेशन ---
console = Console()
try: BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError: BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR); console.print(f"[bold red]चेतावनी: 'data' फोल्डर नहीं मिला! मैंने इसे बना दिया है। कृपया अपनी .txt फाइलें डालें।[/bold red]"); sys.exit(1)

# ==============================================================================
# 1. God-Mode Parser & Feature Engineering
# ==============================================================================

def get_cut(ank): return (ank + 5) % 10

def load_and_engineer_data(filepath):
    """यह पार्सर अब और भी मजबूत है। यह लगभग हर तरह के फॉर्मेट को समझ सकता है।"""
    records = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or '*' in line: continue
                line = line.replace('/', ' ').replace('-', ' ')
                parts = [p.strip() for p in line.split() if p.strip()]
                
                # सबसे महत्वपूर्ण: हम लाइन के अंत से डेटा उठाते हैं।
                # इससे कोई फर्क नहीं पड़ता कि तारीख का फॉर्मेट क्या है।
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
                        'Open_Pana_Digits': {int(d) for d in f"{op_pana:03d}"},
                        'Close_Pana_Digits': {int(d) for d in f"{cl_pana:03d}"},
                    })
        return pd.DataFrame(records) if records else None
    except Exception: return None

# ==============================================================================
# 2. The Synapse Engine: The Brain that Adapts
# ==============================================================================
def generate_hypotheses():
    """हजारों तार्किक संभावनाओं को बनाता है।"""
    source_fields = ['Open', 'Close', 'Open_Cut', 'Close_Cut', 'Jodi_Total', 'Jodi_Diff', 'Open_Pana_Sum', 'Close_Pana_Sum']
    target_fields = ['Open', 'Close', 'Jodi_Total']
    for s_field, t_field in product(source_fields, target_fields):
        if s_field == t_field: continue
        yield {'s': s_field, 't': t_field, 'op': 'equals', 'desc': f"{s_field} -> {t_field}"}
    for p_field in ['Open_Pana_Digits', 'Close_Pana_Digits']:
        for a_field in ['Open', 'Close']:
            yield {'s': p_field, 't': a_field, 'op': 'contains', 'desc': f"{p_field.replace('_Digits','')} -> {a_field}"}

def run_synapse_discovery(all_markets_data, days, attempts_thresh):
    """यह इंजन सभी नियमों को टेस्ट करता है और उनकी सफलता को रिकॉर्ड करता है, चाहे वो पास हों या फेल।"""
    all_tested_patterns = []
    market_names, lags_to_test = list(all_markets_data.keys()), [1, 2, 3, 4, 5, 6, 7]
    hypotheses = list(generate_hypotheses())
    with console.status("[bold green]साइनेप्स ब्रह्मांड को स्कैन कर रहा है...[/bold green]"):
        for m1_name, m2_name in product(market_names, repeat=2):
            for hypo in hypotheses:
                for lag in lags_to_test:
                    df1, df2 = all_markets_data[m1_name].tail(days), all_markets_data[m2_name].tail(days)
                    hits, attempts = test_hypothesis(df1, df2, hypo, lag)
                    if attempts >= attempts_thresh:
                        success_rate = (hits / attempts) * 100
                        all_tested_patterns.append({
                            's_market': m1_name, 't_market': m2_name, 'hypo': hypo, 'lag': lag,
                            'success': success_rate, 'record': f"{hits}/{attempts}", 'hits': hits
                        })
    return all_tested_patterns

def test_hypothesis(df1, df2, hypo, lag):
    hits, attempts = 0, 0
    s_col, t_col, op = hypo['s'], hypo['t'], hypo['op']
    max_len = min(len(df1), len(df2))
    if max_len <= lag: return 0, 0
    for i in range(max_len - lag):
        source_row, target_row = df1.iloc[i], df2.iloc[i + lag]
        if s_col not in source_row or t_col not in target_row: continue
        s_val, t_val = source_row[s_col], target_row[t_col]
        condition_met = (s_val == t_val) if op == 'equals' else (t_val in s_val)
        attempts += 1
        if condition_met: hits += 1
    return hits, attempts

# ==============================================================================
# 3. The Predictor & Reporter: The Voice of the Brain
# ==============================================================================
def generate_and_display_report(target_market, all_patterns, all_markets_data, success_thresh):
    os.system('cls' if os.name == 'nt' else 'clear')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%d-%m-%Y (%A)')

    # --- Adaptive Tier System ---
    # Tier 1: Gold (User's choice)
    hot_rules = [p for p in all_patterns if p['t_market'] == target_market and p['success'] >= success_thresh]
    tier, confidence = "Gold", "[bold green]High[/bold green]"
    
    # Tier 2: Silver (Plan B)
    if not hot_rules:
        silver_thresh = max(60, success_thresh - 10)
        hot_rules = [p for p in all_patterns if p['t_market'] == target_market and p['success'] >= silver_thresh]
        tier, confidence = "Silver", "[bold yellow]Medium[/bold yellow]"

    # Tier 3: Bronze (Most Frequent)
    if not hot_rules:
        all_target_patterns = [p for p in all_patterns if p['t_market'] == target_market]
        hot_rules = sorted(all_target_patterns, key=lambda x: x['hits'], reverse=True)[:15] # Top 15 most frequent
        tier, confidence = "Bronze", "[bold red]Low (Frequency Based)[/bold red]"
        
    # --- Prediction ---
    predictions = defaultdict(lambda: defaultdict(int))
    for rule in hot_rules:
        s_market, lag, hypo = rule['s_market'], rule['lag'], rule['hypo']
        source_df = all_markets_data[s_market]
        if len(source_df) < lag: continue
        s_val = source_df.iloc[-lag][hypo['s']]
        pred_cat = "JODI_TOTAL" if hypo['t'] == "Jodi_Total" else "OTC"
        
        if hypo['op'] == 'equals': predictions[pred_cat][s_val] += rule['success']
        elif hypo['op'] == 'contains': 
            for digit in s_val: predictions[pred_cat][digit] += rule['success']

    final_preds = {}
    if predictions['OTC']:
        sorted_otc = sorted(predictions['OTC'].items(), key=lambda i: i[1], reverse=True)
        final_preds['OTC'] = [ank for ank, score in sorted_otc[:4]]
    if predictions['JODI_TOTAL']:
        sorted_totals = sorted(predictions['JODI_TOTAL'].items(), key=lambda i: i[1], reverse=True)
        if sorted_totals:
            top_total = sorted_totals[0][0]
            jodi_list = {f"{i}{(top_total - i + 10) % 10}" for i in range(10)}
            final_preds['JODI'] = sorted(list(jodi_list))[:12]

    # --- Display Report ---
    header = f"Prediction for {tomorrow}\n"
    header += f"🔥 [bold magenta]{target_market.upper()} - SYNAPSE REPORT[/bold magenta] 🔥\n"
    header += f"[dim]Analysis Tier: [bold]{tier}[/bold] | Prediction Confidence: {confidence}[/dim]"
    console.print(Panel(Text(header, justify="center"), border_style="yellow"))

    if not final_preds:
        console.print(Panel("[bold red]माफ़ करना मास्टर, इस मार्केट के लिए कोई भी उपयोगी पैटर्न नहीं मिला।[/bold red]"))
        return

    table = Table(box=None, padding=(0, 1))
    table.add_column("TYPE", style="bold cyan"); table.add_column("PREDICTION", style="white")
    table.add_row("STRONG OTC", ' - '.join(map(str, final_preds.get('OTC', []))) or "[dim]N/A[/dim]")
    table.add_row("JODI", ' '.join(final_preds.get('JODI', [])) or "[dim]N/A[/dim]")
    console.print(Panel(table, title="[bold]संभावित परिणाम[/bold]", border_style="cyan"))

# ==============================================================================
# 4. Main Controller
# ==============================================================================
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel(Text("🔥 PROJECT SYNAPSE v7.0 - THE LIVING BRAIN 🔥", justify="center", style="bold white on red"),
                        subtitle="[white]मैं सीखने के लिए तैयार हूँ, मास्टर।[/white]", border_style="red"))

    all_data = {}
    with console.status("[green]डेटा फाइलों को स्कैन किया जा रहा है...[/green]"):
        for f in os.listdir(DATA_DIR):
            if f.endswith('.txt'):
                df = load_and_engineer_data(os.path.join(DATA_DIR, f))
                if df is not None and len(df) > 20: all_data[f.replace('.txt','').upper()] = df
    if not all_data:
        console.print("[red]विश्लेषण के लिए कोई योग्य डेटा फाइल नहीं मिली।[/red]"); sys.exit(1)
    console.print(f"[green]✓ साइनेप्स ने {len(all_data)} मार्केट्स को अपनी मेमोरी में लोड कर लिया है।[/green]\n")

    days = int(Prompt.ask("[bold yellow]कितने दिनों के डेटा का विश्लेषण करना है?[/bold yellow]", choices=["30", "60", "90"], default="30"))
    success = int(Prompt.ask("[bold yellow]आपकी पसंदीदा सफलता दर क्या है? (%) [/bold yellow]", default="75"))
    attempts = int(Prompt.ask("[bold yellow]न्यूनतम कितनी बार नियम टेस्ट हुआ हो?[/bold yellow]", default="10"))

    all_patterns = run_synapse_discovery(all_data, days, attempts)
    console.print(f"\n[bold green]✓ विश्लेषण पूरा! साइनेप्स ने {len(all_patterns):,} संभावित पैटर्नों का विश्लेषण किया है।[/bold green]")
    time.sleep(2)

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Panel("[bold]खोजे गए पैटर्न के आधार पर भविष्य का अनुमान लगाएं।[/bold]", title="भविष्यवाणी कंसोल"))
        
        market_names = list(all_data.keys())
        choices = [str(i+1) for i in range(len(market_names))] + ['0']
        
        menu_text = "".join([f"[cyan]{i+1}[/cyan] - {name}\n" for i, name in enumerate(market_names)]) + "\n[red]0[/red] - बाहर निकलें"
        console.print(Panel(menu_text, title="मार्केट चुनें"))

        choice = Prompt.ask("[bold]विकल्प चुनें[/bold]", choices=choices)
        if choice == '0': break

        target_market = market_names[int(choice) - 1]
        
        with console.status(f"[cyan]{target_market} के लिए पैटर्न लागू किए जा रहे हैं...[/cyan]"):
            time.sleep(1) # For dramatic effect
            generate_and_display_report(target_market, all_patterns, all_data, success)
        
        Prompt.ask("\n[yellow]जारी रखने के लिए Enter दबाएं...[/yellow]")

    console.print("\n[bold red]Synapse is offline. अलविदा, मास्टर।[/bold red]")

if __name__ == "__main__":
    main()
