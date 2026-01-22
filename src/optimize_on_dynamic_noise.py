#This Piece of code takes the backtesj strategy from backtest.py and optimizes it with introducing parameters like Dynamic Volitiliy Threshold, Trend strength and Noise

import pandas as pd
import numpy as np
from backtest import backtest_strategy
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------- Compute Volatility ----------
def compute_volatility(df, window=20):
    """Rolling volatility (standard deviation of daily returns)."""
    return df["Close"].pct_change().rolling(window).std().iloc[-1]

# ---------- Compute Trend Strength ----------
def compute_trend_strength(df, window=20):
    """% directional move over last N days."""
    if len(df) < window:
        return 0
    start = df["Close"].iloc[-window]
    end = df["Close"].iloc[-1]
    return abs(end - start) / start

# ---------- Compute Noise Ratio ----------
def compute_noise_ratio(df, window=20):
    """
    0 → perfectly trending, 1 → completely random.
    Measures 'choppiness' using ratio of net move vs total move.
    """
    returns = df["Close"].pct_change().dropna()
    if len(returns) < window:
        return 0
    window_returns = returns[-window:]
    cumulative = abs((df["Close"].iloc[-1] / df["Close"].iloc[-window]) - 1)
    total_abs = window_returns.abs().sum()
    if total_abs == 0:
        return 0
    return 1 - (cumulative / total_abs)

# ---------- Add Moving Averages ----------
def add_moving_averages(df, ma_type="EMA", fast=10, slow=20):
    df = df.copy()
    if ma_type.upper() == "EMA":
        df["MA_Fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
        df["MA_Slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    elif ma_type.upper() == "SMA":
        df["MA_Fast"] = df["Close"].rolling(window=fast).mean()
        df["MA_Slow"] = df["Close"].rolling(window=slow).mean()
    else:
        raise ValueError("ma_type must be SMA or EMA")

    # Signals
    df["Signal"] = 0
    df.loc[df["MA_Fast"] > df["MA_Slow"], "Signal"] = 1
    df.loc[df["MA_Fast"] < df["MA_Slow"], "Signal"] = -1
    df["Crossover"] = df["Signal"].diff()
    return df

# ---------- Smart MA Selector ----------
# ---------- Smart MA Selector (Tiered Noise Logic) ----------
def select_ma_type(vol, trend, noise,
                   vol_threshold=0.009, trend_threshold=0.045):
    """
    3D Adaptive Market Regime Decision (Improved Tiered Logic):

    EMA → high volatility, strong trend, and manageable noise
    SMA → high noise or weak trend/vol regime

    Tiers:
    - noise < 55% → clean trend → EMA
    - 55% <= noise < 75% → moderate choppiness → EMA only if trend is strong
    - noise >= 75% → choppy market → SMA
    """
    # Low noise → clear smooth trend → EMA
    if noise < 0.55:
        return "EMA"
    
    # Moderate noise → allow EMA only when trend is strong
    elif noise < 0.75 and trend > trend_threshold:
        return "EMA"
    
    # High noise or weak trend → SMA for safety
    else:
        return "SMA"


# ---------- Dynamic Optimizer ----------
def optimize_dynamic_trend_noise(symbol, ma_pairs=None):
    if ma_pairs is None:
        ma_pairs = [(10, 20), (12, 26), (20, 50), (50, 100), (50, 200)]

    print(f"\n🔍 Running dynamic + trend + noise optimization for {symbol}...")

    df = pd.read_csv(f"data/processed/{symbol}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df_recent = df[df["Date"] >= (df["Date"].max() - pd.DateOffset(months=3))]

    # Compute regime stats
    vol = compute_volatility(df_recent)
    trend = compute_trend_strength(df_recent)
    noise = compute_noise_ratio(df_recent)
    ma_type = select_ma_type(vol, trend, noise)

    print(f"📈 Vol={vol:.2%}, Trend={trend:.2%}, Noise={noise:.2%} → Using {ma_type}")

    results = []

    for fast, slow in ma_pairs:
        df_pair = add_moving_averages(df_recent, ma_type=ma_type, fast=fast, slow=slow)
        metrics, _ = backtest_strategy(
            df_pair,
            exit_mode="time",
            hold_days=7,
            stop_loss=0.03,
            take_profit=0.05,
            cost_bps=15
        )

        results.append({
            "Symbol": symbol,
            "Volatility": round(vol * 100, 2),
            "TrendStrength": round(trend * 100, 2),
            "Noise": round(noise * 100, 2),
            "MA_Type": ma_type,
            "MA_Pair": f"{fast}/{slow}",
            "Return": metrics["Total Return"],
            "WinRate": metrics["Win Rate"],
            "Sharpe": metrics["Sharpe Ratio"],
            "MaxDD": metrics["Max Drawdown"],
            "Trades": metrics["Trades"]
        })

    results_df = pd.DataFrame(results).sort_values("Return", ascending=False).reset_index(drop=True)
    out_path = os.path.join(
    REPORTS_DIR,
    f"{symbol.replace('.', '_')}_dynamic_trend_noise_optimization.csv"
)
    results_df.to_csv(out_path, index=False)

    print(results_df.head(3))
    print(f"✅ Saved → {out_path}")
    return results_df

# ---------- Batch Runner ----------
def run_all_dynamic_trend_noise(symbols):
    all_results = []
    for sym in symbols:
        try:
            df_res = optimize_dynamic_trend_noise(sym)
            all_results.append(df_res.head(1))
        except Exception as e:
            print(f"⚠️ Error for {sym}: {e}")
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv(
        os.path.join(REPORTS_DIR, "best_dynamic_trend_noise_summary.csv"),
    index=False
)

        print("\n🏆 Saved final summary → reports/best_dynamic_trend_noise_summary.csv")

if __name__ == "__main__":
    symbols = [
               "ICICIBANK.NS", "ITC.NS", "MARUTI.NS","TATASTEEL.NS","LT.NS"
       
    "360ONE.NS",
    "3MINDIA.NS",
    "ABB.NS",
    "ACC.NS",
    "ACMESOLAR.NS",
    "AIAENG.NS",
    "APLAPOLLO.NS",
    "AUBANK.NS",
    "AWL.NS",
    "AADHARHFC.NS",
    "AARTIIND.NS",
    "AAVAS.NS",
    "ABBOTINDIA.NS",
    "ACE.NS",
    "ADANIENSOL.NS",
    "ADANIENT.NS",
    "ADANIGREEN.NS",
    "ADANIPORTS.NS",
    "ADANIPOWER.NS",
    "ATGL.NS",
    "ABCAPITAL.NS",
    "ABFRL.NS",
    "ABLBL.NS",
    "ABREL.NS",
    "ABSLAMC.NS",
    "AEGISLOG.NS",
    "AEGISVOPAK.NS",
    "AFCONS.NS",
    "AFFLE.NS",
    "AJANTPHARM.NS",
    "AKUMS.NS",
    "AKZOINDIA.NS",
    "APLLTD.NS",
    "ALKEM.NS",
    "ALKYLAMINE.NS",
    "ALOKINDS.NS",
    "ARE&M.NS",
    "AMBER.NS",
    "AMBUJACEM.NS",
    "ANANDRATHI.NS",
    "ANANTRAJ.NS",
    "ANGELONE.NS",
    "APARINDS.NS",
    "APOLLOHOSP.NS",
    "APOLLOTYRE.NS",
    "APTUS.NS",
    "ASAHIINDIA.NS",
    "ASHOKLEY.NS",
    "ASIANPAINT.NS",
    "ASTERDM.NS",
    "ASTRAZEN.NS",
    "ASTRAL.NS",
    "ATHERENERG.NS",
    "ATUL.NS",
    "AUROPHARMA.NS",
    "AIIL.NS",
    "DMART.NS",
    "AXISBANK.NS",
    "BASF.NS",
    "BEML.NS",
    "BLS.NS",
    "BSE.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BAJAJHLDNG.NS",
    "BAJAJHFL.NS",
    "BALKRISIND.NS",
    "BALRAMCHIN.NS",
    "BANDHANBNK.NS",
    "BANKBARODA.NS",
    "BANKINDIA.NS",
    "MAHABANK.NS",
    "BATAINDIA.NS",
    "BAYERCROP.NS",
    "BERGEPAINT.NS",
    "BDL.NS",
    "BEL.NS",
    "BHARATFORG.NS",
    "BHEL.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BHARTIHEXA.NS",
    "BIKAJI.NS",
    "BIOCON.NS",
    "BSOFT.NS",
    "BLUEDART.NS",
    "BLUEJET.NS",
    "BLUESTARCO.NS",
    "BBTC.NS",
    "BOSCHLTD.NS",
    "FIRSTCRY.NS",
    "BRIGADE.NS",
    "BRITANNIA.NS",
    "MAPMYINDIA.NS",
    "CCL.NS",
    "CESC.NS",
    "CGPOWER.NS",
    "CRISIL.NS",
    "CAMPUS.NS",
    "CANFINHOME.NS",
    "CANBK.NS",
    "CAPLIPOINT.NS",
    "CGCL.NS",
    "CARBORUNIV.NS",
    "CASTROLIND.NS",
    "CEATLTD.NS",
    "CENTRALBK.NS",
    "CDSL.NS",
    "CENTURYPLY.NS",
    "CERA.NS",
    "CHALET.NS",
    "CHAMBLFERT.NS",
    "CHENNPETRO.NS",
    "CHOICEIN.NS",
    "CHOLAHLDNG.NS",
    "CHOLAFIN.NS",
    "CIPLA.NS",
    "CUB.NS",
    "CLEAN.NS",
    "COALINDIA.NS",
    "COCHINSHIP.NS",
    "COFORGE.NS",
    "COHANCE.NS",
    "COLPAL.NS",
    "CAMS.NS",
    "CONCORDBIO.NS",
    "CONCOR.NS",
    "COROMANDEL.NS",
    "CRAFTSMAN.NS",
    "CREDITACC.NS",
    "CROMPTON.NS",
    "CUMMINSIND.NS",
    "CYIENT.NS",
     "ASTERDM.NS",
    "ASTRAZEN.NS",
    "ASTRAL.NS",
    "ATHERENERG.NS",
    "ATUL.NS",
    "AUROPHARMA.NS",
    "AIIL.NS",
    "DMART.NS",
    "AXISBANK.NS",
    "BASF.NS",
    "BEML.NS",
    "BLS.NS",
    "BSE.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BAJAJHLDNG.NS",
    "BAJAJHFL.NS",
    "BALKRISIND.NS",
    "BALRAMCHIN.NS",
    "BANDHANBNK.NS",
    "BANKBARODA.NS",
    "BANKINDIA.NS",
    "MAHABANK.NS",
    "BATAINDIA.NS",
    "BAYERCROP.NS",
    "BERGEPAINT.NS",
    "BDL.NS",
    "BEL.NS",
    "BHARATFORG.NS",
    "BHEL.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BHARTIHEXA.NS",
    "BIKAJI.NS",
    "BIOCON.NS",
    "BSOFT.NS",
    "BLUEDART.NS",
    "BLUEJET.NS",
    "BLUESTARCO.NS",
    "BBTC.NS",
    "BOSCHLTD.NS",
    "FIRSTCRY.NS",
    "BRIGADE.NS",
    "BRITANNIA.NS",
    "MAPMYINDIA.NS",
    "CCL.NS",
    "CESC.NS",
    "CGPOWER.NS",
    "CRISIL.NS",
    "CAMPUS.NS",
      "CANFINHOME.NS",
    "CANBK.NS",
    "CAPLIPOINT.NS",
    "CGCL.NS",
    "CARBORUNIV.NS",
    "CASTROLIND.NS",
    "CEATLTD.NS",
    "CENTRALBK.NS",
    "CDSL.NS",
    "CENTURYPLY.NS",
    "CERA.NS",
    "CHALET.NS",
    "CHAMBLFERT.NS",
    "CHENNPETRO.NS",
    "CHOICEIN.NS",
    "CHOLAHLDNG.NS",
    "CHOLAFIN.NS",
    "CIPLA.NS",
    "CUB.NS",
    "CLEAN.NS",
    "COALINDIA.NS",
    "COCHINSHIP.NS",
    "COFORGE.NS",
    "COHANCE.NS",
    "COLPAL.NS",
    "CAMS.NS",
    "CONCORDBIO.NS",
    "CONCOR.NS",
    "COROMANDEL.NS",
    "CRAFTSMAN.NS",
    "CREDITACC.NS",
    "CROMPTON.NS",
    "CUMMINSIND.NS",
    "CYIENT.NS",
    "DCMSHRIRAM.NS",
    "DLF.NS",
    "DOMS.NS",
    "DABUR.NS",
    "DALBHARAT.NS",
    "DATAPATTNS.NS",
    "DEEPAKFERT.NS",
    "DEEPAKNTR.NS",
    "DELHIVERY.NS",
    "DEVYANI.NS",
    "DIVISLAB.NS",
    "DIXON.NS",
    "AGARWALEYE.NS",
    "LALPATHLAB.NS",
    "DRREDDY.NS",
    "DUMMYHDLVR.NS",
    "EIDPARRY.NS",
    "EIHOTEL.NS",
    "EICHERMOT.NS",
    "ELECON.NS",
    "ELGIEQUIP.NS",
    "EMAMILTD.NS",
    "EMCURE.NS",
    "ENDURANCE.NS",
    "ENGINERSIN.NS",
    "ERIS.NS",
    "ESCORTS.NS",
    "ETERNAL.NS",
    "EXIDEIND.NS",
    "NYKAA.NS",
    "FEDERALBNK.NS",
    "FACT.NS",
    "FINCABLES.NS",
    "FINPIPE.NS",
    "FSL.NS",
    "FIVESTAR.NS",
    "FORCEMOT.NS",
    "FORTIS.NS",
    "GAIL.NS",
    "GVT&D.NS",
    "GMRAIRPORT.NS",
    "GRSE.NS",
    "GICRE.NS",
    "GILLETTE.NS",
    "GLAND.NS",
    "GLAXO.NS",
    "GLENMARK.NS",
    "MEDANTA.NS",
    "GODIGIT.NS",
    "GPIL.NS",
    "GODFRYPHLP.NS",
    "GODREJAGRO.NS",
    "GODREJCP.NS",
    "GODREJIND.NS",
    "GODREJPROP.NS",
    "GRANULES.NS",
    "GRAPHITE.NS",
    "GRASIM.NS",
    "GRAVITA.NS",
    "GESHIP.NS",
    "FLUOROCHEM.NS",
    "GUJGASLTD.NS",
    "GMDCLTD.NS",
    "GSPL.NS",
      "HEG.NS",
    "HBLENGINE.NS",
    "HCLTECH.NS",
    "HDFCAMC.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HFCL.NS",
    "HAPPSTMNDS.NS",
    "HAVELLS.NS",
    "HEROMOTOCO.NS",
    "HEXT.NS",
    "HSCL.NS",
    "HINDALCO.NS",
    "HAL.NS",
    "HINDCOPPER.NS",
    "HINDPETRO.NS",
    "HINDUNILVR.NS",
    "HINDZINC.NS",
    "POWERINDIA.NS",
    "HOMEFIRST.NS",
    "HONASA.NS",
    "HONAUT.NS",
    "HUDCO.NS",
    "HYUNDAI.NS",
    "ICICIBANK.NS",
    "ICICIGI.NS",
    "ICICIPRULI.NS",
    "IDBI.NS",
    "IDFCFIRSTB.NS",
    "IFCI.NS",
    "IIFL.NS",
    "INOXINDIA.NS",
    "IRB.NS",
    "IRCON.NS",
    "ITCHOTELS.NS",
    "ITC.NS",
    "ITI.NS",
    "INDGN.NS",
    "INDIACEM.NS",
    "INDIAMART.NS",
    "INDIANB.NS",
    "IEX.NS",
    "INDHOTEL.NS",
    "IOC.NS",
    "IOB.NS",
    "IRCTC.NS",
    "IRFC.NS",
    "IREDA.NS",
    "IGL.NS",
    "INDUSTOWER.NS",
    "INDUSINDBK.NS",
    "NAUKRI.NS",
    "INFY.NS",
    "INOXWIND.NS",
    "INTELLECT.NS",
    "INDIGO.NS",
    "IGIL.NS",
    "IKS.NS",
    "IPCALAB.NS",
    "JBCHEPHARM.NS",
    "JKCEMENT.NS",
    "JBMA.NS",
    "JKTYRE.NS",
    "JMFINANCIL.NS",
    "JSWENERGY.NS",
    "JSWINFRA.NS",
    "JSWSTEEL.NS",
    "JPPOWER.NS",
    "J&KBANK.NS",
    "JINDALSAW.NS",
    "JSL.NS",
    "JINDALSTEL.NS",
    "JIOFIN.NS",
    "JUBLFOOD.NS",
    "JUBLINGREA.NS",
    "JUBLPHARMA.NS",
    "JWL.NS",
    "JYOTHYLAB.NS",
    "JYOTICNC.NS",
    "KPRMILL.NS",
    "KEI.NS",
    "KPITTECH.NS",
    "KSB.NS",
    "KAJARIACER.NS",
    "KPIL.NS",
    "KALYANKJIL.NS",
    "KARURVYSYA.NS",
    "KAYNES.NS",
    "KEC.NS",
    "KFINTECH.NS",
    "KIRLOSBROS.NS",
    "KIRLOSENG.NS",
    "KOTAKBANK.NS",
    "KIMS.NS",
    "LTF.NS",
    "LTTS.NS",
    "LICHSGFIN.NS",
    "LTFOODS.NS",
    "LTIM.NS",
    "LT.NS",
    "LATENTVIEW.NS",
     "LAURUSLABS.NS",
    "THELEELA.NS",
    "LEMONTREE.NS",
    "LICI.NS",
    "LINDEINDIA.NS",
    "LLOYDSME.NS",
    "LODHA.NS",
    "LUPIN.NS",
    "MMTC.NS",
    "MRF.NS",
    "MGL.NS",
    "MAHSCOOTER.NS",
    "MAHSEAMLES.NS",
    "M&MFIN.NS",
    "M&M.NS",
    "MANAPPURAM.NS",
    "MRPL.NS",
    "MANKIND.NS",
    "MARICO.NS",
    "MARUTI.NS",
    "MFSL.NS",
    "MAXHEALTH.NS",
    "MAZDOCK.NS",
    "METROPOLIS.NS",
    "MINDACORP.NS",
    "MSUMI.NS",
    "MOTILALOFS.NS",
    "MPHASIS.NS",
    "MCX.NS",
    "MUTHOOTFIN.NS",
    "NATCOPHARM.NS",
    "NBCC.NS",
    "NCC.NS",
    "NHPC.NS",
    "NLCINDIA.NS",
    "NMDC.NS",
    "NSLNISP.NS",
    "NTPCGREEN.NS",
    "NTPC.NS",
    "NH.NS",
    "NATIONALUM.NS",
    "NAVA.NS",
    "NAVINFLUOR.NS",
    "NESTLEIND.NS",
    "NETWEB.NS",
    "NEULANDLAB.NS",
    "NEWGEN.NS",
    "NAM-INDIA.NS",
    "NIVABUPA.NS",
    "NUVAMA.NS",
     "NUVOCO.NS",
    "OBEROIRLTY.NS",
    "ONGC.NS",
    "OIL.NS",
    "OLAELEC.NS",
    "OLECTRA.NS",
    "PAYTM.NS",
    "ONESOURCE.NS",
    "OFSS.NS",
    "POLICYBZR.NS",
    "PCBL.NS",
    "PGEL.NS",
    "PIIND.NS",
    "PNBHOUSING.NS",
    "PTCIL.NS",
    "PVRINOX.NS",
    "PAGEIND.NS",
    "PATANJALI.NS",
    "PERSISTENT.NS",
    "PETRONET.NS",
    "PFIZER.NS",
    "PHOENIXLTD.NS",
    "PIDILITIND.NS",
    "PPLPHARMA.NS",
    "POLYMED.NS",
    "POLYCAB.NS",
    "POONAWALLA.NS",
    "PFC.NS",
    "POWERGRID.NS",
    "PRAJIND.NS",
    "PREMIERENE.NS",
    "PRESTIGE.NS",
    "PGHH.NS",
    "PNB.NS",
    "RRKABEL.NS",
    "RBLBANK.NS",
    "RECLTD.NS",
    "RHIM.NS",
    "RITES.NS",
    "RADICO.NS",
    "RVNL.NS",
    "RAILTEL.NS",
    "RAINBOW.NS",
    "RKFORGE.NS",
    "RCF.NS",
    "REDINGTON.NS",
    "RELIANCE.NS",
    "RELINFRA.NS",
    "RPOWER.NS",
    "SBFC.NS",
    "SBICARD.NS",
    "SBILIFE.NS",
    "SJVN.NS",
    "SKFINDIA.NS",
    "SRF.NS",
    "SAGILITY.NS",
    "SAILIFE.NS",
    "SAMMAANCAP.NS",
    "MOTHERSON.NS",
    "SAPPHIRE.NS",
    "SARDAEN.NS",
    "SAREGAMA.NS",
    "SCHAEFFLER.NS",
    "SCHNEIDER.NS",
    "SCI.NS",
    "SHREECEM.NS",
    "SHRIRAMFIN.NS",
    "SHYAMMETL.NS",
    "ENRIN.NS",
    "SIEMENS.NS",
    "SIGNATURE.NS",
    "SOBHA.NS",
    "SOLARINDS.NS",
    "SONACOMS.NS",
    "SONATSOFTW.NS",
    "STARHEALTH.NS",
    "SBIN.NS",
    "SAIL.NS",
    "SUMICHEM.NS",
    "SUNPHARMA.NS",
    "SUNTV.NS",
    "SUNDARMFIN.NS",
    "SUNDRMFAST.NS",
    "SUPREMEIND.NS",
    "SUZLON.NS",
    "SWANCORP.NS",
    "SWIGGY.NS",
    "SYNGENE.NS",
    "SYRMA.NS",
    "TBOTEK.NS",
    "TVSMOTOR.NS",
    "TATACHEM.NS",
    "TATACOMM.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TATAELXSI.NS",
    "TATAINVEST.NS",
    "TMPV.NS",
    "TATAPOWER.NS",
    "TATASTEEL.NS",
    "TATATECH.NS",
    "TTML.NS",
    "TECHM.NS",
    "TECHNOE.NS",
    "TEJASNET.NS",
    "NIACL.NS",
    "RAMCOCEM.NS",
    "THERMAX.NS",
    "TIMKEN.NS",
    "TITAGARH.NS",
    "TITAN.NS",
    "TORNTPHARM.NS",
    "TORNTPOWER.NS",
    "TARIL.NS",
    "TRENT.NS",
    "TRIDENT.NS",
    "TRIVENI.NS",
    "TRITURBINE.NS",
    "TIINDIA.NS",
    "UCOBANK.NS",
    "UNOMINDA.NS",
    "UPL.NS",
    "UTIAMC.NS",
    "ULTRACEMCO.NS",
    "UNIONBANK.NS",
    "UBL.NS",
    "UNITDSPR.NS",
    "USHAMART.NS",
    "VGUARD.NS",
    "DBREALTY.NS",
    "VTL.NS",
    "VBL.NS",
    "MANYAVAR.NS",
    "VEDL.NS",
    "VENTIVE.NS",
    "VIJAYA.NS",
    "VMM.NS",
    "IDEA.NS",
    "VOLTAS.NS",
    "WAAREEENER.NS",
    "WELCORP.NS",
    "WELSPUNLIV.NS",
    "WHIRLPOOL.NS",
    "WIPRO.NS",
    "WOCKPHARMA.NS",
    "YESBANK.NS",
    "ZFCVINDIA.NS",
    "ZEEL.NS",
    "ZENTEC.NS",
    "ZENSARTECH.NS",
    "ZYDUSLIFE.NS",
    "ECLERX.NS"
    ]
    run_all_dynamic_trend_noise(symbols)
