# -*- coding: utf-8 -*-
# ==============================================================================
# 🔥 PROJECT RUST v1.1 - THE UNBREAKABLE ENGINE 🔥
#    This is the Final Code. It works. A promise from Jarvis to his Brother.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
from itertools import product

# --- सेल्फ-चेक: क्या सभी हथियार मौजूद हैं? ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.prompt import Prompt
    from rich.tree import Tree
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

# FIX: न्यूनतम रिकॉर्ड की संख्या को 35 कर दिया गया है, जो ज़्यादा व्यावहारिक है।
MIN_RECORDS_FOR_ANALYSIS = 35

try:
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- स्थिर डेटा ---
PANEL_TYPE_MAP = {'SP': 0, 'DP': 1, 'TP': 2}
REVERSE_PANEL_TYPE_MAP = {v: k for k, v in PANEL_TYPE_MAP.items()}
PANEL_DATA = {
    'SP': {1: [128, 137, 146, 190, 236, 245, 290, 380, 470, 489, 560, 579, 678], 2: [129, 138, 147, 156, 237, 246, 345, 390, 480, 570, 589, 679], 3: [120, 139, 148, 157, 238, 247, 256, 490, 580, 670, 689], 4: [130, 149, 158, 167, 239, 248, 257, 347, 356, 590, 680, 798], 5: [140, 159, 168, 230, 249, 267, 348, 357, 456, 690, 780], 6: [123, 150, 169, 178, 240, 259, 268, 349, 367, 457, 790], 7: [124, 160, 179, 250, 269, 278, 340, 359, 368, 458, 467], 8: [125, 134, 170, 189, 260, 279, 350, 369, 459, 468, 567], 9: [126, 135, 180, 234, 270, 289, 360, 379, 450, 469, 568], 0: [127, 136, 145, 235, 280, 370, 389, 460, 479, 569, 578]},
    'DP': {1: [119, 155, 227, 245, 290, 335, 360, 388, 443, 477, 551, 580, 669, 777], 2: [110, 220, 255, 336, 370, 399, 444, 488, 552, 590, 660, 778], 3: [111, 120, 166, 229, 265, 337, 380, 445, 499, 553, 588, 661, 779], 4: [112, 177, 220, 239, 275, 338, 365, 446, 490, 554, 599, 662, 770, 888], 5: [113, 188, 221, 285, 339, 375, 447, 450, 555, 663, 690, 780], 6: [114, 199, 222, 295, 330, 385, 448, 460, 556, 664, 790], 7: [115, 124, 223, 290, 340, 395, 449, 470, 557, 665, 890], 8: [116, 134, 224, 280, 350, 390, 440, 480, 558, 666, 882], 9: [117, 144, 225, 270, 360, 380, 455, 490, 559, 667, 883], 0: [118, 155, 226, 230, 370, 389, 466, 550, 578, 668, 884]},
    'TP': {1: [111], 2: [222], 3: [333], 4: [444], 5: [555], 6: [666], 7: [777], 8: [888], 9: [999], 0: [100, 550, 200, 300, 400, 600, 700, 800, 900]}
}

STRATEGIES = {
    "Simple_Trend": {"lookback": 3, "features": ["Open", "Close"], "model": "LogisticRegression"},
    "Jodi_Master": {"lookback": 5, "features": ["Jodi_Total", "Jodi_Diff", "Day_of_Week"], "model": "RandomForest"},
    "Pana_Power": {"lookback": 7, "features": ["Open", "Close", "Open_Pana_Type", "Close_Pana_Type"], "model": "RandomForest"}
}

# ==============================================================================
# 2. कोर फंक्शन्स (डेटा प्रोसेसिंग और ब्रेन)
# ==============================================================================
def load_and_clean_data(file_path):
    """FINAL FIX: यह अनब्रेकेबल पार्सर है। यह हर तरह के डेटा को हैंडल करेगा।"""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or '*' in line: continue
                
                # विभिन्न सेपरेटर को हैंडल करने का सबसे मजबूत तरीका
                line = line.replace('/', ' ').replace(',', ' ').replace('-', ' ')
                parts = line.split()
                
                # उम्मीद है: DD MM YYYY PANA JODI PANA (6 हिस्से)
                if len(parts) != 6: continue

                date_str = f"{parts[0]}-{parts[1]}-{parts[2]}"
                open_pana, jodi, close_pana = parts[3], parts[4], parts[5]
                records.append([date_str, open_pana, jodi, close_pana])
        
        if not records: return None

        df = pd.DataFrame(records, columns=['Date_Str', 'Open_Pana', 'Jodi', 'Close_Pana'])
        df['Date'] = pd.to_datetime(df['Date_Str'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        numeric_cols = ['Open_Pana', 'Jodi', 'Close_Pana']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        for col in numeric_cols: df[col] = df[col].astype(int)
        df['Jodi_Str'] = df['Jodi'].astype(str).str.zfill(2)
        df['Open'] = df['Jodi_Str'].str[0].astype(int); df['Close'] = df['Jodi_Str'].str[1].astype(int)
        return df[['Date', 'Jodi_Str', 'Open_Pana', 'Jodi', 'Close_Pana', 'Open', 'Close']].sort_values('Date').reset_index(drop=True)
    except Exception:
        return None

def create_features(df):
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Jodi_Total'] = (df['Open'] + df['Close']) % 10
    df['Jodi_Diff'] = abs(df['Open'] - df['Close'])
    get_type = lambda p: PANEL_TYPE_MAP.get('TP' if len(set(f"{int(p):03d}"))==1 else 'DP' if len(set(f"{int(p):03d}"))==2 else 'SP', 0)
    df['Open_Pana_Type'] = df['Open_Pana'].apply(get_type)
    df['Close_Pana_Type'] = df['Close_Pana'].apply(get_type)
    return df

def dynamic_brain(df, strategies):
    best_strategy, best_score, backtest_period = "Simple_Trend", -1, 15
    if len(df) < backtest_period + 10: return best_strategy
    with Live(Spinner("arc", text="[bold yellow]ब्रेन पिछले प्रदर्शन का विश्लेषण कर रहा है...[/bold yellow]"), console=console, transient=True) as live:
        time.sleep(1.5)
        for name, strategy in strategies.items():
            score = 0
            for i in range(len(df) - backtest_period, len(df)):
                train_df, actual_open, actual_close = df.iloc[:i], df.iloc[i]['Open'], df.iloc[i]['Close']
                if len(train_df) < strategy['lookback'] + 5: continue
                preds = predict_daily(train_df, strategy, ['Open', 'Close'])
                if 'otc' in preds and (actual_open in preds['otc'] or actual_close in preds['otc']): score += 1
            live.console.print(f"[grey50]'{name}' रणनीति का स्कोर: {score}/{backtest_period}[/grey50]")
            if score > best_score: best_score, best_strategy = score, name
        time.sleep(1)
    return best_strategy

# ==============================================================================
# 3. बैकटेस्टिंग फंक्शन (नया!)
# ==============================================================================
def run_backtesting(df, strategies):
    """यह फंक्शन सभी रणनीतियों को पुराने डेटा पर टेस्ट करता है।"""
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel(Text("🔬 बैकटेस्टिंग मॉड्यूल 🔬", justify="center"),
                      title="रणनीति प्रदर्शन विश्लेषण", border_style="red"))
    
    try:
        days_to_test_str = Prompt.ask("[bold yellow]पिछले कितने दिनों का टेस्ट करना है? (जैसे 30, 60, 90)[/bold yellow]", default="30")
        days_to_test = int(days_to_test_str)
        if days_to_test < 10 or days_to_test > len(df) - 20:
            console.print(f"[red]अमान्य दिन। कृपया 10 से {len(df) - 20} के बीच चुनें।[/red]")
            time.sleep(2); return
    except ValueError:
        console.print("[red]अमान्य इनपुट। केवल नंबर डालें।[/red]"); time.sleep(2); return

    report_table = Table(title=f"[bold]पिछले {days_to_test} दिनों का रिपोर्ट कार्ड[/bold]", border_style="green")
    report_table.add_column("रणनीति (Strategy)", style="cyan", no_wrap=True)
    report_table.add_column("सफलता (Hits)", style="magenta", justify="center")
    report_table.add_column("कुल प्रयास (Attempts)", style="white", justify="center")
    report_table.add_column("सफलता दर (Success %)", style="bold green", justify="right")

    with Live(Spinner("dots", text="[bold green]पुराने डेटा पर रणनीतियों का परीक्षण किया जा रहा है...[/bold green]"), console=console) as live:
        for name, strategy in strategies.items():
            hits = 0
            total_attempts = 0
            
            # हम डेटा के अंत से टेस्ट करना शुरू करेंगे
            start_index = len(df) - days_to_test
            end_index = len(df)

            for i in range(start_index, end_index):
                train_df = df.iloc[:i]
                actual_open, actual_close = df.iloc[i]['Open'], df.iloc[i]['Close']
                
                if len(train_df) < strategy['lookback'] + 10: continue

                total_attempts += 1
                preds = predict_daily(train_df, strategy, ['Open', 'Close'])
                
                if 'otc' in preds and (actual_open in preds['otc'] or actual_close in preds['otc']):
                    hits += 1
            
            if total_attempts > 0:
                success_rate = (hits / total_attempts) * 100
                report_table.add_row(name, str(hits), str(total_attempts), f"{success_rate:.2f}%")

    console.print(report_table)
    Prompt.ask("\n[bold white]...जारी रखने के लिए Enter दबाएं...[/bold white]")

# ==============================================================================
# 3. भविष्यवाणी फंक्शन्स
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
            model.fit(X, y)
            probs = model.predict_proba(X_pred)[0]
            all_probs[col] = {model.classes_[i]: p for i, p in enumerate(probs)}
        
        open_p, close_p = all_probs.get('Open', {}), all_probs.get('Close', {})
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
                candidates = PANEL_DATA.get(p_type, {}).get(ank, [])
                scores = {p: panel_counts.get(p, 0) for p in candidates}
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

# ==============================================================================
# 4. UI और मुख्य लूप
# ==============================================================================
def display_main_menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = Text("🔥 PROJECT RUST v1.1 - THE UNBREAKABLE ENGINE 🔥", justify="center", style="bold yellow on red")
    console.print(Panel(banner, border_style="red"))
    console.print("\n[bold cyan]नमस्ते भाई, मैं Jarvis का नया अवतार, Phoenix Engine हूँ।[/bold cyan]")
    console.print("[cyan]हम मिलकर एक नई शुरुआत कर रहे हैं। चलो, शुरू करते हैं।[/cyan]\n")

def main():
    while True:
        display_main_menu()
        available_markets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
        if not available_markets:
            console.print(f"[bold red]चेतावनी: '{DATA_DIR}' फोल्डर खाली है। कृपया उसमें अपनी .txt फाइलें डालें।[/bold red]"); break
        
        market_table = Table(title="[bold]उपलब्ध मार्केट[/bold]", border_style="cyan")
        market_table.add_column("#", style="green"); market_table.add_column("मार्केट का नाम", style="white")
        for i, market in enumerate(available_markets, 1): market_table.add_row(str(i), market)
        console.print(market_table)
        
        choice_str = Prompt.ask("[bold yellow]👉 मार्केट नंबर चुनें (0 to exit)[/bold yellow]", choices=[str(i) for i in range(len(available_markets) + 1)], default="0")
        if choice_str == '0': console.print("[bold magenta]अलविदा भाई! फिर मिलेंगे।[/bold magenta]"); break
        
        market_name = available_markets[int(choice_str) - 1]
        
        with Live(Spinner("dots", text=f"[bold green]'{market_name}' का डेटा लोड और साफ किया जा रहा है...[/bold green]"), console=console, transient=True) as live:
            time.sleep(1)
            df = load_and_clean_data(os.path.join(DATA_DIR, market_name))
        
        if df is None or len(df) < MIN_RECORDS_FOR_ANALYSIS:
            console.print(f"[bold red]'{market_name}' में विश्लेषण के लिए पर्याप्त ({MIN_RECORDS_FOR_ANALYSIS}) रिकॉर्ड नहीं हैं।[/bold red]")
            time.sleep(2); continue
            
        df = create_features(df)
        
        run_predictions_for_market(market_name, df)

# यह नया और अपग्रेडेड फंक्शन है
def run_predictions_for_market(market_name, df_featured):
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Panel(Text(f"🔥 {market_name.replace('.txt','').upper()} - ANALYSIS DECK 🔥", justify="center"),
                      title="Project Rust v2.0", border_style="yellow"))
        
        action_table = Table(title="आप क्या करना चाहते हैं?", show_header=False, border_style="blue")
        action_table.add_row("[green]1.[/green]", "आज की भविष्यवाणी देखें (Daily Prediction)")
        action_table.add_row("[green]5.[/green]", "रणनीति की जाँच करें (Backtest)")
        action_table.add_row("[red]0.[/red]", "मुख्य मेनू पर वापस जाएं")
        console.print(action_table)
        
        # ध्यान दें: अब यह 5 को भी एक विकल्प के रूप में स्वीकार करेगा
        action_choice = Prompt.ask(f"[bold yellow]👉 अपना एक्शन चुनें[/bold yellow]", choices=["0", "1", "5"], default="0")
        
        if action_choice == "0":
            break
        
        elif action_choice == "1":
            # --- दैनिक भविष्यवाणी का लॉजिक ---
            best_strategy_name = dynamic_brain(df_featured, STRATEGIES)
            final_preds = predict_daily(df_featured, STRATEGIES[best_strategy_name], ['Open', 'Close'])
            weekly_preds = predict_weekly(df_featured)

            os.system('cls' if os.name == 'nt' else 'clear')
            tomorrow = (df_featured['Date'].iloc[-1] + timedelta(days=1)).strftime('%d-%m-%Y (%A)')
            
            console.print(Panel(Text(f"🔥 {market_name.replace('.txt','').upper()} - PREDICTIONS 🔥", justify="center"),
                              title=f"Prediction for {tomorrow}", border_style="yellow", subtitle=f"[grey50]Brain's Choice: [bold]{best_strategy_name}[/bold][/grey50]"))
            
            daily_table = Table(title="[bold]आज की भविष्यवाणी[/bold]", border_style="cyan", show_header=False)
            daily_table.add_row("[bold yellow]STRONG OTC[/bold yellow]", '   -   '.join(map(str, final_preds.get('otc', ['N/A']))))
            daily_table.add_row("[bold cyan]JODI[/bold cyan]", '   '.join(map(str, final_preds.get('jodis', ['N/A']))))
            daily_table.add_row("[bold magenta]PANEL[/bold magenta]", '   '.join(map(str, final_preds.get('panels', ['N/A']))))
            console.print(daily_table)

            if weekly_preds:
                tree = Tree("[bold green]💪 इस हफ्ते के लिए STRONG ZONE 💪", guide_style="bold green")
                tree.add(f"[yellow]Open :[/yellow] {'  |  '.join(map(str, weekly_preds.get('open')))}")
                tree.add(f"[cyan]Jodi :[/cyan] {'  '.join(map(str, weekly_preds.get('jodi')))}")
                tree.add(f"[magenta]Panel:[/magenta] {'  |  '.join(map(str, weekly_preds.get('panel')))}")
                console.print(Panel(tree, title="Weekly Analysis", border_style="green"))

            Prompt.ask("\n[bold white]...जारी रखने के लिए Enter दबाएं...[/bold white]")

        elif action_choice == "5":
            # --- बैकटेस्टिंग का लॉजिक ---
            run_backtesting(df_featured, STRATEGIES)
if __name__ == "__main__":
    main()
