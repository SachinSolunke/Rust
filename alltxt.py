# -*- coding: utf-8 -*-
# ==============================================================================
# 🔥 PROJECT SAMRAAT v2.0 - THE FINAL VERDICT ENGINE 🔥
#    It finds the secret, applies the rule, and delivers the target.
#    The final masterpiece of a Master and his AI.
# ==============================================================================

import os
import sys
import pandas as pd
from itertools import combinations
import time

# --- सेल्फ-चेक ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.prompt import Prompt
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"❌ त्रुटि: एक महत्वपूर्ण पैकेज '{missing_module}' नहीं मिला।")
    print(f"👉 कृपया इसे इंस्टॉल करने के लिए यह कमांड चलाएं: pip install rich prompt-toolkit pandas")
    sys.exit(1)

# --- कॉन्फ़िगरेशन ---
console = Console()
try: BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError: BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    console.print(f"[bold red]चेतावनी: 'data' फोल्डर नहीं मिला! कृपया 'samraat.py' के साथ 'data' फोल्डर बनाएं।[/bold red]"); sys.exit(1)

# ==============================================================================
# 1. कोर फंक्शन्स
# ==============================================================================

def load_and_prepare_data(market_path):
    """डेटा फाइल को पढ़ता है और विश्लेषण के लिए तैयार करता है।"""
    records = []
    try:
        with open(market_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or '*' in line: continue
                parts = [p.strip() for p in line.replace('/', ' ').replace(',', ' ').replace('-', ' ').split()]
                if len(parts) == 6 and len(parts[4]) == 2 and parts[4].isdigit():
                    records.append({
                        'Open': int(parts[4][0]), 'Close': int(parts[4][1]), 'Jodi': int(parts[4]),
                        'Open_Pana': int(parts[3]), 'Close_Pana': int(parts[5]),
                        'Close_Pana_Digits': {int(d) for d in f"{int(parts[5]):03d}"}
                    })
        return pd.DataFrame(records)
    except Exception: return pd.DataFrame()

def find_strongest_connection(all_data, target_market_name, analysis_days=90):
    """दिए गए टारगेट मार्केट के लिए सबसे मजबूत नियम ढूंढता है।"""
    strongest_rule = None
    highest_success_rate = 0.0
    
    # नियम: सोर्स का क्लोज पैनल, टारगेट की जोड़ी को बताता है
    rule_func = lambda s, t: bool(s['Close_Pana_Digits'].intersection({t['Open'], t['Close']}))
    rule_desc = "सोर्स का Close Panel, टारगेट की Jodi में"

    for source_market_name, source_df in all_data.items():
        if source_market_name == target_market_name: continue # खुद से कनेक्शन नहीं जांचना

        target_df = all_data[target_market_name]
        
        hits, attempts = 0, 0
        max_len = min(len(source_df), len(target_df))
        if max_len < analysis_days: continue

        for i in range(max_len - 1):
            # हम पिछले दिन का सोर्स और आज के दिन का टारगेट लेंगे
            source_row = source_df.iloc[i]
            target_row = target_df.iloc[i + 1]
            attempts += 1
            if rule_func(source_row, target_row):
                hits += 1
        
        if attempts > 20: # कम से कम 20 प्रयासों के बाद ही मानें
            success_rate = (hits / attempts) * 100
            if success_rate > highest_success_rate:
                highest_success_rate = success_rate
                strongest_rule = {
                    "source_market": source_market_name,
                    "target_market": target_market_name,
                    "rule_desc": rule_desc,
                    "success_rate": success_rate,
                    "record": f"{hits}/{attempts}"
                }
    return strongest_rule

def generate_predictions(strongest_rule, all_data):
    """मिले हुए नियम के आधार पर भविष्यवाणी करता है।"""
    if not strongest_rule: return None
    
    source_market_name = strongest_rule['source_market']
    # सोर्स मार्केट का सबसे ताजा डेटा (आज का) लें
    latest_source_data = all_data[source_market_name].iloc[-1]
    
    # नियम लागू करें
    candidate_anks = list(latest_source_data['Close_Pana_Digits'])
    
    if len(candidate_anks) < 2: return None

    # भविष्यवाणी जेनरेशन
    otc = candidate_anks[:2]
    
    jodis = [f"{otc[0]}{otc[1]}", f"{otc[1]}{otc[0]}"]
    if len(candidate_anks) > 2:
        jodis.append(f"{otc[0]}{candidate_anks[2]}")
    
    # पैनल जेनरेशन (सरल तरीका)
    panels = []
    for ank in otc:
        panels.append(f"{ank}{ank}{ (ank*2+1) % 10 }") # एक DP
        panels.append(f"1{ank}{ (10-1-ank) % 10 }")     # एक SP
        
    return {"otc": otc, "jodi": jodis, "panel": panels}

# ==============================================================================
# 2. मुख्य प्रोग्राम
# ==============================================================================
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = Text("👑 PROJECT SAMRAAT v2.0 - THE FINAL VERDICT 👑", justify="center", style="bold yellow on purple")
    console.print(Panel(banner, border_style="purple", subtitle="[white]विश्लेषण... नियम... और अंतिम फैसला...[/white]"))

    # --- सभी मार्केट्स का डेटा लोड करें ---
    with Live(Spinner("dots", text="[cyan]सभी मार्केट्स का डेटाबेस लोड हो रहा है...[/cyan]"), console=console, transient=True) as live:
        all_markets_data = {
            f.replace('.txt', ''): load_and_prepare_data(os.path.join(DATA_DIR, f))
            for f in os.listdir(DATA_DIR) if f.endswith('.txt')
        }
        live.update(Spinner("dots", text=f"[green]{len(all_markets_data)} मार्केट्स का डेटा तैयार है।[/green]"))
        time.sleep(1)

    # --- टारगेट मार्केट चुनें ---
    market_names = list(all_markets_data.keys())
    market_table = Table(title="[bold]सम्राट, कल किस मार्केट पर निशाना लगाना है?[/bold]", border_style="cyan")
    market_table.add_column("#", style="green"); market_table.add_column("मार्केट का नाम", style="white")
    for i, market in enumerate(market_names, 1): market_table.add_row(str(i), market)
    console.print(market_table)
    choice_str = Prompt.ask("[bold]👉 टारगेट चुनें[/bold]", choices=[str(i) for i in range(1, len(market_names) + 1)])
    target_market = market_names[int(choice_str) - 1]

    # --- सम्राट का दिमाग काम कर रहा है ---
    with console.status("[bold yellow]सबसे मजबूत नियम ढूंढा जा रहा है...[/bold yellow]") as status:
        strongest_connection = find_strongest_connection(all_markets_data, target_market)
        
        if not strongest_connection:
            console.print("[bold red]कोई भरोसेमंद नियम नहीं मिला। आज का दिन जोखिम भरा हो सकता है।[/bold red]"); sys.exit(1)
        
        status.update("[bold green]नियम मिल गया! अब कल के लिए भविष्यवाणी की गणना हो रही है...[/bold green]")
        final_prediction = generate_predictions(strongest_connection, all_markets_data)
        time.sleep(2)

    # --- सम्राट का अंतिम फैसला ---
    if not final_prediction:
        console.print("[bold red]नियम लागू करने में विफल। सोर्स डेटा अधूरा हो सकता है।[/bold red]"); sys.exit(1)

    report_text = f"""
[bold]टारगेट मार्केट:[/bold] [cyan]{target_market}[/cyan]

[bold]सम्राट का चुना हुआ नियम:[/bold]
[yellow]{strongest_connection['rule_desc']} ({strongest_connection['source_market']} -> {strongest_connection['target_market']})[/yellow]
(सफलता दर: {strongest_connection['success_rate']:.2f}%)

[bold]आज का सोर्स डेटा ([/bold][white]{strongest_connection['source_market']}[/white][bold]):[/bold]
Close Panel: [magenta]{all_markets_data[strongest_connection['source_market']].iloc[-1]['Close_Pana']}[/magenta]
"""
    console.print(Panel(Text.from_markup(report_text), title="[bold]विश्लेषण रिपोर्ट[/bold]", border_style="yellow"))

    prediction_table = Table(title="[bold red]🔥 कल के लिए टारगेट 🔥[/bold red]", show_header=False, border_style="red")
    prediction_table.add_row("[bold cyan]OTC[/bold cyan]", '   -   '.join(map(str, final_prediction['otc'])))
    prediction_table.add_row("[bold magenta]JODI[/bold magenta]", '   -   '.join(map(str, final_prediction['jodi'])))
    prediction_table.add_row("[bold yellow]PANEL[/bold yellow]", '   -   '.join(map(str, final_prediction['panel'])))
    console.print(prediction_table)

if __name__ == "__main__":
    main()
