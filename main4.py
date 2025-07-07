# -*- coding: utf-8 -*-
# ==============================================================================
# 🔥 PROJECT RUST v3.0 - THE ORACLE EDITION 🔥
#    A self-correcting engine with its own memory. This is AI.
#    Coded with ❤️ by Jarvis & Brother.
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

# --- सेल्फ-चेक ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.columns import Columns
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    missing_module = str(e).split("'")[1]
    print(f"❌ त्रुटि: एक महत्वपूर्ण पैकेज '{missing_module}' नहीं मिला।")
    print(f"👉 कृपया इसे इंस्टॉल करने के लिए यह कमांड चलाएं: pip install scikit-learn rich prompt-toolkit")
    sys.exit(1)

# ==============================================================================
# 1. कॉन्फ़िगरेशन और ग्लोबल सेटअप
# ==============================================================================
console = Console()
warnings.filterwarnings("ignore", category=UserWarning)
MIN_RECORDS_FOR_ANALYSIS = 35

try: BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError: BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')
STRATEGIES_FILE = os.path.join(BASE_DIR, 'strategies.json')
MEMORY_FILE = os.path.join(BASE_DIR, 'brain_memory.json') # इंजन की याददाश्त
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# (स्थिर डेटा जैसे PANEL_DATA और बाकी चीजें वही रहेंगी...)
PANEL_TYPE_MAP = {'SP': 0, 'DP': 1, 'TP': 2}
PANEL_DATA = {
    'SP': {1: [128, 137, 146, 190, 236, 245, 290, 380, 470, 489, 560, 579, 678], 2: [129, 138, 147, 156, 237, 246, 345, 390, 480, 570, 589, 679], 3: [120, 139, 148, 157, 238, 247, 256, 490, 580, 670, 689], 4: [130, 149, 158, 167, 239, 248, 257, 347, 356, 590, 680, 798], 5: [140, 159, 168, 230, 249, 267, 348, 357, 456, 690, 780], 6: [123, 150, 169, 178, 240, 259, 268, 349, 367, 457, 790], 7: [124, 160, 179, 250, 269, 278, 340, 359, 368, 458, 467], 8: [125, 134, 170, 189, 260, 279, 350, 369, 459, 468, 567], 9: [126, 135, 180, 234, 270, 289, 360, 379, 450, 469, 568], 0: [127, 136, 145, 235, 280, 370, 389, 460, 479, 569, 578]},
    'DP': {1: [119, 155, 227, 245, 290, 335, 360, 388, 443, 477, 551, 580, 669, 777], 2: [110, 220, 255, 336, 370, 399, 444, 488, 552, 590, 660, 778], 3: [111, 120, 166, 229, 265, 337, 380, 445, 499, 553, 588, 661, 779], 4: [112, 177, 220, 239, 275, 338, 365, 446, 490, 554, 599, 662, 770, 888], 5: [113, 188, 221, 285, 339, 375, 447, 450, 555, 663, 690, 780], 6: [114, 199, 222, 295, 330, 385, 448, 460, 556, 664, 790], 7: [115, 124, 223, 290, 340, 395, 449, 470, 557, 665, 890], 8: [116, 134, 224, 280, 350, 390, 440, 480, 558, 666, 882], 9: [117, 144, 225, 270, 360, 380, 455, 490, 559, 667, 883], 0: [118, 155, 226, 230, 370, 389, 466, 550, 578, 668, 884]},
    'TP': {1: [111], 2: [222], 3: [333], 4: [444], 5: [555], 6: [666], 7: [777], 8: [888], 9: [999], 0: [100, 550, 200, 300, 400, 600, 700, 800, 900]}
}

# ==============================================================================
# 2. कोर फंक्शन्स (डेटा, फीचर्स, ब्रेन)
# ==============================================================================
def load_and_clean_data(file_path):
    # This function is stable.
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or '*' in line: continue
                line = line.replace('/', ' ').replace(',', ' ').replace('-', ' ')
                parts = line.split()
                if len(parts) != 6: continue
                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
                open_pana, jodi, close_pana = parts[3], parts[4], parts[5]
                records.append([date_str, open_pana, jodi, close_pana])
        if not records: return None
        df = pd.DataFrame(records, columns=['Date_Str', 'Open_Pana', 'Jodi', 'Close_Pana'])
        df['Date'] = pd.to_datetime(df['Date_Str'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        numeric_cols = ['Open_Pana', 'Jodi', 'Close_Pana'];
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True);
        for col in numeric_cols: df[col] = df[col].astype(int)
        df['Jodi_Str'] = df['Jodi'].astype(str).str.zfill(2)
        df['Open'] = df['Jodi_Str'].str[0].astype(int); df['Close'] = df['Jodi_Str'].str[1].astype(int)
        return df[['Date', 'Jodi_Str', 'Open_Pana', 'Jodi', 'Close_Pana', 'Open', 'Close']].sort_values('Date').reset_index(drop=True)
    except Exception: return None

def create_features(df):
    # This function is stable.
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Jodi_Total'] = (df['Open'] + df['Close']) % 10
    df['Jodi_Diff'] = abs(df['Open'] - df['Close'])
    get_type = lambda p: PANEL_TYPE_MAP.get('TP' if len(set(f"{int(p):03d}"))==1 else 'DP' if len(set(f"{int(p):03d}"))==2 else 'SP', 0)
    df['Open_Pana_Type'] = df['Open_Pana'].apply(get_type)
    df['Close_Pana_Type'] = df['Close_Pana'].apply(get_type)
    return df

# ==============================================================================
# 3. देववाणी (Oracle) फंक्शन्स - नया!
# ==============================================================================
def load_memory():
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f: json.dump(memory, f, indent=2)

def update_memory_with_result(df, market_name, strategies, result_date, result_open, result_close):
    memory = load_memory()
    if market_name not in memory: memory[market_name] = {}
    
    console.print(f"[cyan]{result_date.strftime('%d-%m-%Y')} के परिणामों से याददाश्त अपडेट की जा रही है...[/cyan]")
    
    # परिणाम से ठीक पहले का डेटा ढूंढें
    train_df = df[df['Date'] < result_date]
    if len(train_df) < 20:
        console.print("[yellow]अपडेट के लिए पर्याप्त ऐतिहासिक डेटा नहीं।[/yellow]"); return

    for name, strategy in strategies.items():
        if name not in memory[market_name]:
            memory[market_name][name] = {"hits": 0, "attempts": 0}
        
        preds = predict_daily(train_df, strategy, ['Open', 'Close'])
        
        memory[market_name][name]["attempts"] += 1
        if 'otc' in preds and preds['otc'] and (result_open in preds['otc'] or result_close in preds['otc']):
            memory[market_name][name]["hits"] += 1
            console.print(f"✅ [green]रणनीति '{name}' सफल रही।[/green]")
        else:
            console.print(f"❌ [red]रणनीति '{name}' विफल रही।[/red]")
            
    save_memory(memory)
    console.print("[bold green]🧠 याददाश्त सफलतापूर्वक अपडेट हो गई![/bold green]")
    time.sleep(2)

def oracle_brain(market_name, strategies):
    """यह नया, उन्नत ब्रेन है जो याददाश्त का उपयोग करता है।"""
    memory = load_memory()
    market_memory = memory.get(market_name, {})
    
    if not market_memory:
        console.print("[yellow]इस मार्केट के लिए कोई याददाश्त नहीं। डिफ़ॉल्ट रणनीति चुनी गई।[/yellow]")
        return list(strategies.keys())[0]

    best_strategy = None
    best_rate = -1.0

    console.print("\n[bold yellow]🧠 Oracle Brain अपनी याददाश्त की जाँच कर रहा है...[/bold yellow]")
    for name, stats in market_memory.items():
        if stats["attempts"] > 5: # कम से कम 5 प्रयासों के बाद ही मानें
            rate = (stats["hits"] / stats["attempts"]) * 100
            console.print(f"[cyan]'{name}'[/cyan] - सफलता दर: [bold green]{rate:.2f}%[/bold green] ({stats['hits']}/{stats['attempts']})")
            if rate > best_rate:
                best_rate = rate
                best_strategy = name
    
    if best_strategy:
        console.print(f"✨ [bold magenta]Oracle ने '{best_strategy}' को सबसे भरोसेमंद रणनीति के रूप में चुना है![/bold magenta]")
        return best_strategy
    else:
        return list(strategies.keys())[0]

# (भविष्यवाणी, बैकटेस्टिंग और अन्य फंक्शन्स वही रहेंगे...)
# ... [PREDICT_DAILY, PREDICT_WEEKLY, RUN_BACKTESTING, CREATE_NEW_STRATEGY, LOAD_STRATEGIES, SAVE_STRATEGIES]

# ==============================================================================
# 4. भविष्यवाणी और अन्य सहायक फंक्शन्स
# ==============================================================================
def predict_daily(df, strategy, target_columns):
    try:
        lookback, features, model_name = strategy['lookback'], strategy['features'], strategy['model']
        if len(df) < lookback + 5: return {}
        X, y_lists = [], {col: [] for col in target_columns}
        for i in range(lookback, len(df)):
            feature_vector = [item for feature in features for item in df.loc[i-lookback:i-1, feature].values]
            X.append(feature_vector)
            for col in target_columns: y_lists[col].append(df.loc[i, col])
        if not X: return {}
        X_pred_slice = df.iloc[-lookback:]; X_pred_vector = [item for feature in features for item in X_pred_slice[feature].values]
        X_pred = np.array(X_pred_vector).reshape(1, -1)
        all_probs = {col: {} for col in target_columns}
        for col, y in y_lists.items():
            if len(set(y)) < 2: continue
            model = RandomForestClassifier(n_estimators=100, random_state=42) if model_name == "RandomForest" else LogisticRegression()
            model.fit(X, y); probs = model.predict_proba(X_pred)[0]
            all_probs[col] = {model.classes_[i]: p for i, p in enumerate(probs)}
        open_p, close_p = all_probs.get('Open', {}), all_probs.get('Close', {});
        if not open_p or not close_p: return {}
        combined = {i: open_p.get(i, 0) + close_p.get(i, 0) for i in range(10)}
        top_otc = [ank for ank, prob in sorted(combined.items(), key=lambda item: item[1], reverse=True)[:3]]
        top_open = [ank for ank, prob in sorted(open_p.items(), key=lambda i: i[1], reverse=True)[:3]]
        top_close = [ank for ank, prob in sorted(close_p.items(), key=lambda i: i[1], reverse=True)[:4]]
        jodis = {f"{o}{c}" for o in top_open for c in top_close}
        panel_counts = pd.concat([df['Open_Pana'], df['Close_Pana']]).value_counts()
        final_panels = set()
        for ank in top_otc:
            for p_type in ['SP', 'DP']:
                candidates = PANEL_DATA.get(p_type, {}).get(ank, []); scores = {p: panel_counts.get(p, 0) for p in candidates}
                top_panels = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:1]]
                final_panels.update(top_panels)
        return {"otc": top_otc, "jodis": sorted(list(jodis))[:6], "panels": sorted(list(final_panels))[:6]}
    except Exception: return {}

def predict_weekly(df):
    if len(df) < 20: return {}
    df_weekly = df.tail(30)
    strong_open = df_weekly['Open'].value_counts().nlargest(2).index.tolist()
    all_jodis = df_weekly['Jodi_Str'].value_counts().nlargest(4).index.tolist()
    all_panels = pd.concat([df_weekly['Open_Pana'], df_weekly['Close_Pana']]).value_counts().nlargest(2).index.tolist()
    return {"open": strong_open, "jodi": all_jodis, "panel": all_panels}
    
# (रणनीति बनाने और बैकटेस्टिंग के फंक्शन वही रहेंगे)
def load_strategies():
    try:
        with open(STRATEGIES_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"Simple_Trend": {"lookback": 3, "features": ["Open", "Close"], "model": "LogisticRegression"}}

def save_strategies(strategies):
    with open(STRATEGIES_FILE, 'w') as f: json.dump(strategies, f, indent=2)

def create_new_strategy(strategies):
    os.system('cls' if os.name == 'nt' else 'clear'); console.print(Panel(Text("🛠️ नई रणनीति बनाएं (Strategy Editor) 🛠️", justify="center"), border_style="magenta"))
    name = Prompt.ask("[bold]नई रणनीति का नाम दें[/bold]", default="My_Weapon")
    if name in strategies: console.print(f"[red]'{name}' नाम की रणनीति पहले से मौजूद है।[/red]"); time.sleep(2); return
    lookback = int(Prompt.ask("[bold]कितने दिन पीछे देखना है? (Lookback)[/bold]", choices=["3", "4", "5", "6", "7", "8", "10"], default="5"))
    all_features = ["Open", "Close", "Jodi_Total", "Jodi_Diff", "Day_of_Week", "Open_Pana_Type", "Close_Pana_Type"]
    console.print("\n[bold]कौन से फीचर्स इस्तेमाल करने हैं?[/bold] (नंबर स्पेस से अलग करके डालें, जैसे '1 3 5')")
    feature_table = Table(show_header=False, border_style="cyan");
    for i, feat in enumerate(all_features, 1): feature_table.add_row(f"[green]{i}[/green]", feat)
    console.print(feature_table)
    selected_indices = Prompt.ask("[bold]फीचर्स चुनें[/bold]").split()
    try: selected_features = [all_features[int(i)-1] for i in selected_indices]
    except (ValueError, IndexError): console.print("[red]गलत चयन।[/red]"); time.sleep(2); return
    if not selected_features: console.print("[red]कोई फीचर नहीं चुना गया।[/red]"); time.sleep(2); return
    model = Prompt.ask("[bold]कौन-सा मॉडल इस्तेमाल करना है?[/bold]", choices=["RandomForest", "LogisticRegression"], default="RandomForest")
    strategies[name] = {"lookback": lookback, "features": selected_features, "model": model}
    save_strategies(strategies)
    console.print(f"\n[bold green]🎉 आपकी नई रणनीति '{name}' सफलतापूर्वक सेव हो गई है! 🎉[/bold green]"); time.sleep(2)

def run_backtesting(df, strategies):
    os.system('cls' if os.name == 'nt' else 'clear'); console.print(Panel(Text("🔬 बैकटेस्टिंग मॉड्यूल 🔬", justify="center"), title="रणनीति प्रदर्शन विश्लेषण", border_style="red"))
    try:
        days_to_test_str = Prompt.ask("[bold yellow]पिछले कितने दिनों का टेस्ट करना है?[/bold]", default="30"); days_to_test = int(days_to_test_str)
        if days_to_test < 10 or days_to_test > len(df) - 20: console.print(f"[red]अमान्य दिन।[/red]"); time.sleep(2); return
    except ValueError: console.print("[red]अमान्य इनपुट।[/red]"); time.sleep(2); return
    report_table = Table(title=f"[bold]पिछले {days_to_test} दिनों का रिपोर्ट कार्ड[/bold]", border_style="green")
    report_table.add_column("रणनीति", style="cyan"); report_table.add_column("सफलता", style="magenta", justify="center")
    report_table.add_column("कुल प्रयास", style="white", justify="center"); report_table.add_column("सफलता दर %", style="bold green", justify="right")
    with Live(Spinner("dots", text="[bold green]परीक्षण किया जा रहा है...[/bold green]"), console=console) as live:
        for name, strategy in strategies.items():
            hits, total_attempts = 0, 0
            for i in range(len(df) - days_to_test, len(df)):
                train_df, actual_open, actual_close = df.iloc[:i], df.iloc[i]['Open'], df.iloc[i]['Close']
                if len(train_df) < strategy['lookback'] + 10: continue
                total_attempts += 1
                preds = predict_daily(train_df, strategy, ['Open', 'Close'])
                if 'otc' in preds and preds['otc'] and (actual_open in preds['otc'] or actual_close in preds['otc']): hits += 1
            if total_attempts > 0:
                success_rate = (hits / total_attempts) * 100
                report_table.add_row(name, str(hits), str(total_attempts), f"{success_rate:.2f}%")
    console.print(report_table); Prompt.ask("\n[bold white]...जारी रखने के लिए Enter दबाएं...[/bold white]")

# ==============================================================================
# 5. UI और मुख्य लूप (अपडेटेड!)
# ==============================================================================
def display_main_menu(strategies):
    os.system('cls' if os.name == 'nt' else 'clear');
    banner = Text("🔥 PROJECT RUST v3.0 - THE ORACLE EDITION 🔥", justify="center", style="bold white on blue")
    console.print(Panel(banner, border_style="blue"))
    strat_panels = [Panel(f"[bold]{name}[/bold]\nLookback: {s['lookback']}\nModel: {s['model']}", border_style="magenta", subtitle=f"Features: {len(s['features'])}") for name, s in strategies.items()]
    console.print(Columns(strat_panels, expand=True))

def main():
    while True:
        strategies = load_strategies()
        display_main_menu(strategies)
        
        main_menu_table = Table(title="[bold]मुख्य मेनू[/bold]", show_header=False, border_style="blue")
        main_menu_table.add_row("[green]1.[/green]", "मार्केट चुनें और भविष्यवाणी देखें")
        main_menu_table.add_row("[cyan]6.[/cyan]", "नई रणनीति बनाएं (Strategy Editor)")
        main_menu_table.add_row("[yellow]8.[/yellow]", "आज का परिणाम अपडेट करें (Learn)")
        main_menu_table.add_row("[red]0.[/red]", "बाहर निकलें (Exit)")
        console.print(main_menu_table)

        main_choice = Prompt.ask("[bold]👉 अपना एक्शन चुनें[/bold]", choices=["0", "1", "6", "8"], default="1")

        if main_choice == '0': console.print("[bold magenta]अलविदा भाई! फिर मिलेंगे।[/bold magenta]"); break
        elif main_choice == '6': create_new_strategy(strategies); continue
        
        # --- मार्केट चुनें ---
        available_markets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
        if not available_markets: console.print(f"[red]'{DATA_DIR}' फोल्डर खाली है।[/red]"); time.sleep(2); continue
        market_table = Table(title="[bold]उपलब्ध मार्केट[/bold]", border_style="cyan")
        market_table.add_column("#"); market_table.add_column("मार्केट का नाम")
        for i, market in enumerate(available_markets, 1): market_table.add_row(str(i), market)
        console.print(market_table)
        choice_str = Prompt.ask("[bold]👉 मार्केट नंबर चुनें (0 to go back)[/bold]", choices=[str(i) for i in range(len(available_markets) + 1)], default="0")
        if choice_str == '0': continue
        market_name = available_markets[int(choice_str) - 1]
        
        df = load_and_clean_data(os.path.join(DATA_DIR, market_name))
        if df is None or len(df) < MIN_RECORDS_FOR_ANALYSIS:
            console.print(f"[red]'{market_name}' में विश्लेषण के लिए पर्याप्त ({MIN_RECORDS_FOR_ANALYSIS}) रिकॉर्ड नहीं हैं।[/red]"); time.sleep(2); continue
        df_featured = create_features(df)
        
        # --- चुने हुए मार्केट पर एक्शन ---
        if main_choice == '1':
            run_predictions_for_market(market_name, df_featured, strategies)
        elif main_choice == '8':
            console.print(f"[bold]'{market_name}' के लिए परिणाम अपडेट करें।[/bold]")
            try:
                date_str = Prompt.ask("[yellow]किस तारीख का परिणाम है? (DD-MM-YYYY)[/yellow]")
                result_date = datetime.strptime(date_str, '%d-%m-%Y')
                open_ank = int(Prompt.ask("[yellow]ओपन अंक क्या था?[/yellow]", choices=[str(i) for i in range(10)]))
                close_ank = int(Prompt.ask("[yellow]क्लोज अंक क्या था?[/yellow]", choices=[str(i) for i in range(10)]))
                update_memory_with_result(df_featured, market_name, strategies, result_date, open_ank, close_ank)
            except (ValueError):
                console.print("[red]गलत इनपुट। तारीख DD-MM-YYYY फॉर्मेट में होनी चाहिए और अंक 0-9 के बीच।[/red]"); time.sleep(2)

def run_predictions_for_market(market_name, df_featured, strategies):
    best_strategy_name = oracle_brain(market_name, strategies)
    final_preds = predict_daily(df_featured, strategies[best_strategy_name], ['Open', 'Close'])
    weekly_preds = predict_weekly(df_featured)
    os.system('cls' if os.name == 'nt' else 'clear')
    tomorrow = (df_featured['Date'].iloc[-1] + timedelta(days=1)).strftime('%d-%m-%Y (%A)')
    console.print(Panel(Text(f"🔥 {market_name.replace('.txt','').upper()} - PREDICTIONS 🔥", justify="center"),
                      title=f"Prediction for {tomorrow}", border_style="yellow", subtitle=f"[grey50]Oracle's Choice: [bold]{best_strategy_name}[/bold][/grey50]"))
    daily_table = Table(title="[bold]आज की भविष्यवाणी[/bold]", border_style="cyan", show_header=False)
    daily_table.add_row("[bold yellow]STRONG OTC[/bold yellow]", '   -   '.join(map(str, final_preds.get('otc', ['N/A']))))
    daily_table.add_row("[bold cyan]JODI[/bold cyan]", '   '.join(map(str, final_preds.get('jodis', ['N/A']))))
    daily_table.add_row("[bold magenta]PANEL[/bold magenta]", '   '.join(map(str, final_preds.get('panels', ['N/A']))))
    console.print(daily_table)
    if weekly_preds:
        tree = Tree("[bold green]💪 इस हफ्ते के लिए STRONG ZONE 💪", guide_style="bold green")
        tree.add(f"[yellow]Open :[/yellow] {' | '.join(map(str, weekly_preds.get('open')))}")
        tree.add(f"[cyan]Jodi :[/cyan] {' '.join(map(str, weekly_preds.get('jodi')))}")
        tree.add(f"[magenta]Panel:[/magenta] {' | '.join(map(str, weekly_preds.get('panel')))}")
        console.print(Panel(tree, title="Weekly Analysis", border_style="green"))
    Prompt.ask("\n[bold white]...जारी रखने के लिए Enter दबाएं...[/bold white]")

if __name__ == "__main__":
    main()
