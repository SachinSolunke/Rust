# 🔥 PROJECT PHOENIX v5.0 - THE ULTIMATE FUSION 🔥
#    The Rule-Discovery power of Deep Dive, fused with the unbreakable UI of Rust.
#    A new beginning, built from the best of both worlds, for my Master.
# ==============================================================================

import os
import sys
import pandas as pd
from itertools import combinations, product
import time

# --- सेल्फ-चेक: क्या सभी हथियार मौजूद हैं? ---
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

# ==============================================================================
# 1. कॉन्फ़िगरेशन और ग्लोबल सेटअप
# ==============================================================================
console = Console()
try: BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError: BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

if not os.path.exists(DATA_DIR):
    console.print(f"[bold red]चेतावनी: 'data' फोल्डर नहीं मिला! मैंने इसे बना दिया है।[/bold red]")
    console.print(f"[yellow]कृपया अपनी सभी .txt फाइलें इस फोल्डर में डालें: {DATA_DIR}[/yellow]")
    os.makedirs(DATA_DIR)
    sys.exit(1)

# ==============================================================================
# 2. कोर फंक्शन्स (डेटा प्रोसेसिंग और कनेक्शन टेस्टिंग)
# ==============================================================================

def load_all_market_data(min_records=20):
    """'data' फोल्डर से सभी मार्केट फाइलों को पढ़ता है और उन्हें तैयार करता है।"""
    all_data = {}
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    if not files: return None

    with console.status("[bold green]सभी मार्केट्स का डेटा लोड और तैयार किया जा रहा है...[/bold green]"):
        for filename in files:
            market_name = filename.replace('.txt', '').upper()
            filepath = os.path.join(DATA_DIR, filename)
            records = []
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or '*' in line: continue
                        parts = [p.strip() for p in line.replace('/', ' ').replace(',', ' ').replace('-', ' ').split()]
                        if len(parts) == 6 and len(parts[4]) == 2 and parts[4].isdigit():
                            records.append({
                                'Open': int(parts[4][0]), 'Close': int(parts[4][1]),
                                'Open_Pana_Digits': {int(d) for d in f"{int(parts[3]):03d}"},
                                'Jodi_Total': (int(parts[4][0]) + int(parts[4][1])) % 10
                            })
                if len(records) >= min_records:
                    all_data[market_name] = pd.DataFrame(records)
            except Exception as e:
                console.print(f"[red]'{filename}' को लोड करने में त्रुटि: {e}[/red]")
    return all_data

def test_connection(df1, df2, rule_func, lag_days):
    """दो मार्केट्स के बीच एक खास नियम को दिए गए दिन के गैप पर टेस्ट करता है।"""
    hits, attempts = 0, 0
    max_len = min(len(df1), len(df2))
    if max_len <= lag_days: return 0, 0
    for i in range(max_len - lag_days):
        source_row, target_row = df1.iloc[i], df2.iloc[i + lag_days]
        attempts += 1
        if rule_func(source_row, target_row):
            hits += 1
    return hits, attempts

# ==============================================================================
# 3. नियम और पैटर्न (The Rulebook) - आपके मेनू के अनुसार
# ==============================================================================

# नियम 1 और 2: "3 OTC" (क्या सोर्स का क्लोज, टारगेट के ओपन या क्लोज में आता है?)
rule_otc = lambda s, t: s['Close'] == t['Open'] or s['Close'] == t['Close']
rule_otc_desc = "3 OTC (Close se Open/Close)"

# नियम 3 और 4: "4 Jodi" (क्या सोर्स की जोड़ी का टोटल, टारगेट की जोड़ी का टोटल है?)
rule_jodi = lambda s, t: s['Jodi_Total'] == t['Jodi_Total']
rule_jodi_desc = "4 Jodi (Total se Total)"

# नियम 5 और 6: "6 Panel" (क्या सोर्स के ओपन पैनल का कोई अंक, टारगेट की जोड़ी में आता है?)
rule_panel = lambda s, t: bool(s['Open_Pana_Digits'].intersection({t['Open'], t['Close']}))
rule_panel_desc = "6 Panel (Pana se Jodi)"

# ==============================================================================
# 4. विश्लेषण और UI
# ==============================================================================

def run_analysis(rule_func, rule_desc, all_markets_data, market_choices=None):
    """मुख्य विश्लेषण इंजन। यह दिए गए नियम को सभी या चुने हुए मार्केट्स पर चलाता है।"""
    strong_connections = []
    
    # तय करें कि किन मार्केट्स पर विश्लेषण करना है
    if market_choices:
        markets_to_run = [market_choices]
    else:
        market_names = list(all_markets_data.keys())
        markets_to_run = list(combinations(market_names, 2))

    # हम 1, 2, और 3 दिन के गैप को टेस्ट करेंगे
    lags_to_test = [1, 2, 3]
    total_checks = len(markets_to_run) * len(lags_to_test)
    
    with console.status(f"[bold yellow]विश्लेषण जारी है... नियम: '{rule_desc}'[/bold yellow]") as status:
        count = 0
        for m_pair in markets_to_run:
            for lag in lags_to_test:
                count += 1
                m1_name, m2_name = m_pair[0], m_pair[1]
                status.update(f"[yellow]जाँच हो रही है [{count}/{total_checks}]: {m1_name} -> {m2_name} ({lag} दिन गैप)[/yellow]")
                
                df1, df2 = all_markets_data[m1_name], all_markets_data[m2_name]
                hits, attempts = test_connection(df1, df2, rule_func, lag)

                if attempts > 10: # कम से कम 10 बार टेस्ट हुआ हो
                    success_rate = (hits / attempts) * 100
                    if success_rate >= 60: # सिर्फ 60% से ज़्यादा सफल कनेक्शन
                        strong_connections.append({
                            "Connection": f"{m1_name} -> {m2_name}",
                            "Rule": f"{rule_desc} ({lag} दिन गैप)",
                            "Success": f"{success_rate:.2f}%",
                            "Record": f"{hits}/{attempts}",
                            "SortKey": success_rate
                        })
    return strong_connections

def choose_markets(all_markets_data):
    """यूजर को मार्केट चुनने का विकल्प देता है।"""
    market_names = list(all_markets_data.keys())
    
    console.print("\n[bold]उपलब्ध मार्केट्स:[/bold]")
    table = Table(show_header=False, border_style="cyan")
    table.add_column("#", style="green")
    table.add_column("मार्केट", style="white")
    for i, name in enumerate(market_names, 1):
        table.add_row(str(i), name)
    console.print(table)
    
    choices = [str(i) for i in range(1, len(market_names) + 1)]
    source_num_str = Prompt.ask("[bold yellow]सोर्स मार्केट का नंबर चुनें[/bold yellow]", choices=choices)
    target_num_str = Prompt.ask("[bold yellow]टारगेट मार्केट का नंबर चुनें[/bold yellow]", choices=choices)
    
    source_market = market_names[int(source_num_str) - 1]
    target_market = market_names[int(target_num_str) - 1]
    
    console.print(f"\n[green]👍 चुना गया कनेक्शन: [bold]{source_market} -> {target_market}[/bold][/green]")
    time.sleep(1.5)
    return source_market, target_market

def display_report(connections, rule_desc):
    """विश्लेषण के बाद मिली रिपोर्ट को सुन्दर टेबल में दिखाता है।"""
    console.print("\n\n")
    if not connections:
        console.print(Panel(f"[bold yellow]इस विश्लेषण ('{rule_desc}') में कोई मजबूत कनेक्शन नहीं मिला।[/bold yellow]",
                              title="कोई परिणाम नहीं", border_style="red"))
        return

    report_table = Table(title=f"[bold]विश्लेषण रिपोर्ट: {rule_desc}[/bold]", border_style="green")
    report_table.add_column("कनेक्शन", style="cyan", width=30)
    report_table.add_column("कामयाब नियम (गैप)", style="white", width=35)
    report_table.add_column("सफलता दर", style="bold green", justify="right")
    report_table.add_column("रिकॉर्ड", style="magenta", justify="center")

    sorted_connections = sorted(connections, key=lambda x: x['SortKey'], reverse=True)
    for conn in sorted_connections:
        report_table.add_row(conn["Connection"], conn["Rule"], conn["Success"], conn["Record"])
    console.print(report_table)

def display_main_menu():
    """यूजर को दिखने वाला मेनू प्रिंट करता है।"""
    menu_text = """[bold]🔥Welcome to Market - Tool🔥[/bold]
------------------------------------
[bold cyan]1)[/bold cyan] All Market - 3 OTC
[bold yellow]2)[/bold yellow] Choice You - 3 OTC

[bold cyan]3)[/bold cyan] All Market - 4 Jodi
[bold yellow]4)[/bold yellow] Choice You - 4 Jodi

[bold cyan]5)[/bold cyan] All Market - 6 Panel
[bold yellow]6)[/bold yellow] Choice You - 6 Panel
------------------------------------
[bold red]0) EXIT[/bold red]
"""
    console.print(Panel(menu_text, title="[white]Project Phoenix v5.0[/white]", border_style="purple", expand=False))

# ==============================================================================
# 5. मुख्य लूप
# ==============================================================================
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel(Text("🔥 PROJECT PHOENIX v5.0 - THE ULTIMATE FUSION 🔥", justify="center", style="bold white on purple"),
                        border_style="purple", subtitle="[white]नमस्ते मास्टर, मैं आपके लिए तैयार हूँ...[/white]"))
    
    all_markets_data = load_all_market_data()
    if not all_markets_data:
        console.print(f"[bold red]'{DATA_DIR}' फोल्डर में कोई भी योग्य डेटा फाइल नहीं मिली। प्रोग्राम बंद हो रहा है।[/bold red]")
        sys.exit(1)
        
    console.print(f"[bold green]✓ विश्लेषण के लिए {len(all_markets_data)} मार्केट्स का डेटा सफलतापूर्वक लोड हो गया है।[/bold green]\n")
    time.sleep(2)

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        display_main_menu()
        choice = Prompt.ask("\n[bold]अपना विकल्प चुनें[/bold]", choices=["1", "2", "3", "4", "5", "6", "0"], default="1")
        
        connections = []
        market_selection = None
        
        if choice == '1':
            connections = run_analysis(rule_otc, rule_otc_desc, all_markets_data)
        elif choice == '2':
            market_selection = choose_markets(all_markets_data)
            connections = run_analysis(rule_otc, rule_otc_desc, all_markets_data, market_choices=market_selection)
        elif choice == '3':
            connections = run_analysis(rule_jodi, rule_jodi_desc, all_markets_data)
        elif choice == '4':
            market_selection = choose_markets(all_markets_data)
            connections = run_analysis(rule_jodi, rule_jodi_desc, all_markets_data, market_choices=market_selection)
        elif choice == '5':
            connections = run_analysis(rule_panel, rule_panel_desc, all_markets_data)
        elif choice == '6':
            market_selection = choose_markets(all_markets_data)
            connections = run_analysis(rule_panel, rule_panel_desc, all_markets_data, market_choices=market_selection)
        elif choice == '0':
            console.print("\n[bold green]अलविदा, मास्टर! जल्द मिलेंगे।[/bold green]")
            break
        
        selected_rule_desc = {
            '1': rule_otc_desc, '2': rule_otc_desc, '3': rule_jodi_desc,
            '4': rule_jodi_desc, '5': rule_panel_desc, '6': rule_panel_desc
        }.get(choice)

        display_report(connections, selected_rule_desc)
        Prompt.ask("\n[yellow]आगे बढ़ने के लिए Enter दबाएं...[/yellow]")

if __name__ == "__main__":
    main()
