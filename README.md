# Return Prediction (ML) — README

Projekt buduje model klasyfikacji binarnej przewidujący prawdopodobieństwo zwrotu zamówienia (`Returned ∈ {0,1}`) na podstawie danych transakcyjnych. Pipeline obejmuje: wczytanie danych, preprocessing, inżynierię cech na poziomie transakcji, walidację (10-fold CV), hold-out test, strojenie XGBoost (Optuna) oraz generowanie wykresów wyników modeli.

## Wymagania
- Python 3.10+ (zalecane 3.11)
- System: Windows/Linux/macOS
- Pakiety zainstalowane w virtualenv

## Struktura projektu
```
retrun_pred_project/
├─ data/
│  └─ order_dataset.csv
├─ outputs/
│  ├─ best_xgb_params.json
│  └─ models/                  # wykresy ROC/PR/CM/feature importance
├─ src/
│  ├─ config.py                # stałe i konfiguracja (ścieżki, random_state)
│  ├─ data_loader.py           # wczytanie danych i walidacja wejścia
│  ├─ preprocessing.py         # audit jakości, duplikaty, czyszczenie
│  ├─ feature_engineering.py   # agregacja do poziomu transakcji + cechy
│  ├─ models.py                # definicje modeli (RF, LogReg, XGB)
│  ├─ train.py                 # trening + metryki na hold-out
│  ├─ cv.py                    # 10-fold stratified CV i metryki
│  ├─ tuning.py                # Optuna tuning dla XGBoost
│  └─ model_viz.py             # wykresy ROC/PR/CM/feature importance
├─ tests/                      # testy jednostkowe/integracyjne (pytest)
├─ main.py                     # główne uruchomienie pipeline'u
├─ run_eda.py                  # uruchomienie EDA (opcjonalnie)
└─ run_sanity_check.py         # sanity check danych/cech (opcjonalnie)
```

## 1) Pobranie repozytorium
Jeśli masz repo na GitHubie:
```bash
git clone <URL_DO_REPO>
cd retrun_pred_project
```

Jeśli pracujesz lokalnie — przejdź do katalogu projektu:
```bash
cd D:\Users\micha\Desktop\retrun_pred_project
```

## 2) Utworzenie środowiska i instalacja zależności

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Następnie zainstaluj wymagane biblioteki (jeśli masz `requirements.txt`):
```bash
pip install -r requirements.txt
```

Jeśli nie masz `requirements.txt`, minimalny zestaw (dostosuj do projektu):
```bash
pip install pandas numpy scikit-learn matplotlib xgboost optuna pytest
```

## 3) Dane wejściowe
W katalogu `data/` musi znajdować się plik:
```
data/order_dataset.csv
```

Jeśli go nie ma, pobierz go i umieść w `data/` (np. z Kaggle).

## 4) Konfiguracja ścieżki do danych
W `src/config.py` powinna być ustawiona ścieżka:
- albo względna: `DATA_PATH = "data/order_dataset.csv"`
- albo absolutna (jeśli tak wolisz)

Upewnij się, że wskazuje na istniejący plik.

## 5) Uruchomienie pipeline'u i wygenerowanie wyników
Główne uruchomienie:
```bash
python main.py
```

Pipeline wykonuje:
- wczytanie danych (`data_loader.py`)
- preprocessing i audit (`preprocessing.py`)
- budowę cech na poziomie transakcji (`feature_engineering.py`)
- podział hold-out 80/20
- 10-fold CV dla modeli (LogReg, RandomForest, XGBoost)
- tuning XGBoost (Optuna) i zapis parametrów do `outputs/best_xgb_params.json`
- ewaluację hold-out dla XGBoost przed/po tuningu
- generowanie wykresów do `outputs/models/`

Po udanym uruchomieniu powinny pojawić się m.in.:
- `outputs/best_xgb_params.json`
- `outputs/models/roc_holdout.png`
- `outputs/models/pr_holdout.png`
- `outputs/models/cm_*.png`
- `outputs/models/fi_rf.png`
- `outputs/models/fi_xgb_tuned.png`

## 6) EDA (opcjonalnie)
Jeśli masz przygotowany skrypt EDA:
```bash
python run_eda.py
```
Wyniki (wykresy) powinny zostać zapisane do katalogu `figures/` lub `outputs/` (w zależności od implementacji).


## 7) Testy (pytest)
Uruchom wszystkie testy:
```bash
pytest -q
```

Jeśli pojawi się błąd `ModuleNotFoundError: No module named 'src'`, uruchamiaj testy z katalogu głównego projektu albo ustaw `PYTHONPATH`.

Windows (PowerShell):
```powershell
$env:PYTHONPATH="."
pytest -q
```

Linux/macOS:
```bash
PYTHONPATH=. pytest -q
```

## Typowe problemy i szybkie rozwiązania
- `ConvergenceWarning` w LogisticRegression: to ostrzeżenie o zbieżności; zwykle pomaga standaryzacja cech lub zwiększenie `max_iter`. Nie blokuje uruchomienia projektu.
- Bardzo wysoka ważność `Transaction ID`: nie używaj `Transaction ID` jako cechy wejściowej (to identyfikator, nie cecha), bo może fałszować interpretację i uczyć model „artefaktów”.
- Niezbalansowane dane: wyniki accuracy mogą być mylące. Patrz głównie na ROC-AUC/PR-AUC/F1 i dobór progu.

## Reprodukowalność
W projekcie używany jest `random_state=42` (w konfiguracji oraz w splitach), co ułatwia powtarzalność wyników.
