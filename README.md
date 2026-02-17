# 🚶🏽‍♂️ UR Fall Detection Classic ML
Projekt przedstawia system do detekcji upadków na podstawie wyznaczonych (przy pomocy frameworka [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html)) punktów kluczowych w poszczególnych klatkach wideo wykorzystując do klasyfikacji metody klasycznego uczenia maszynowego. <br>
W celu realizacji systemu zastosowano **klasyfikacje dwuetapową** wykorzystującą dwa wytrenowane modele.

## 📚 O projekcie
Danymi wykorzystanymi w projekcie były dane z datasetu [UR Fall Detection](https://fenix.ur.edu.pl/mkepski/ds/uf.html). Projekt zawiera:
- przygotowanie danych do uczenia z sekwencji wideo z wykorzystaniem **MMPose**
- preprocessing danych
- walidacje wybranych modeli klasycznego uczenia maszynowego
- pipeline pełnego systemu do detekcji upadków wytrenowanego na datasecie **UR Fall Detection**

## 🧰 Kluczowe wymagania

- Python 3.10.18
- [PyTorch 2.5.1](https://pytorch.org)
- [CUDA 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- MMCV 2.1.0
> Opis odtworzenia środowiska znajduje się poniżej w podpunkcie [Ustawienie środowiska](#env)

## 🔸 Opis projektu

### 1.  Przygotowanie danych 

Dane zostały przygotowane przy pomocy narzędzia **MMPose** wykorzystując jako detektor obiektów model **Faster R-CNN** oraz do estymacji punktów kluczowych model **HRNet-32**, oba te modele trenowane były na zbiorze COCO. <br>

W pierwszym kroku sekwencje klatek zostały połączone w pliki wideo, które mogły zostać przetworzone za pomocą **MMPose** w tym celu użyto biblioteki [ffmpeg](https://www.ffmpeg.org).<br>

W kolejnym kroku sekwencje wideo zostały wyznaczone punkty kluczowe w danej sekwencji wideo za pomocą polecenia:<br>
```
python demo/topdown_demo_with_mmdet.py
demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py
checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth configs/tdhm_hrnet-w32_8xb64-210e_coco-256x192.py checkpoints/td-hm_hrnet-w32_8xb64-
210e_coco-256x192-81c58e40_20220909.pth --input ‘INPUT-PATH’ --output-root
‘RESULTS’ --save-predictions
```
> Jako ‘INPUT-PATH’ należy podać ścieżkę do badanego pliku MP4 jako wyjście
otrzymamy w folderze ‘RESULTS’ plik MP4 z wizualizacją keypointów oraz plik JSON
z zapisanymi danymi z detektora, który jest plikiem docelowym zawierającym potrzebne dane.

📒 Realizacja punktu znajduje się w folderze 'scripts' w pliku Jupyter Notebook - ***detect_videos.ipynb***

### 2. Preprocessing i agregacja

1. W pierwszym kroku dla każdej klatki z uzyskanych danych została przeprowadzona:
    - normalizacja danych poprzez odjęcie punktu pozycji środka biodra jako punktu odniesienia
    - wyznaczenie wysokości postaci i podzielenie wyniku przez otrzymany wynik
    - przypisanie etykiety dla każdej klatki na podstawie pliku CSV z datasetu
    - usunięcie klatek zawierających etykiete reprezentującą moment upadku
    - podzielenie danych w formie klatek sekwencjami

Tak przygotowane dane mogły zostać użyte do walidacji oraz trenowania **pierwszego modelu**, który służył jako **klasyfikator pozy** z danej klatki. Model mógł klasyfikować czy w danej sekwencji postać przyjmuje pozycję leżącą czy nie.

2. W drugim kroku przygotowane wcześniej dane zostały poddane dalszemu preprocessingowi:
    - sekwencje w których wcale nie występowała pozycja leżąca zostały odrzucone aby uniknąć szumu w danych
    - na pozostałych sekwencjach przeprowadzono agregacje danych w której do reprezentacji wideo obliczono metryki:
      - **Prędkość** (średnia, minimum, maksimum)
      - **Przyspieszenie** (maksimum)
      - **Pozycje każdej współrzędnej** (średnia, minimum, maksimum, odchylenie standardowe)
    - sekwencje oetykietowano na podstawie tego czy w danej sekwencji występuje **FALL** (upadek) czy **ADL** (czynność codzienna)

Dzięki agregacji uzyskane zostały pojedyncze wektory reprezentujące każdą pełną sekwencje i co się w niej działo na podstawie wyliczonych metryk.<br>
Dane po preprocessingu i agregacji można było wykorzystać do uczenia **drugiego modelu**, którego zadaniem była **klasyfikacja zdarzenia** czy postać w danej sekwencji upada czy wykonuje czynność codzienną (np. kładzie się na łóżko).

📒 Realizacja punktu znajduje się w folderze 'scripts' w pliku Jupyter Notebook - ***preprocessing.ipynb***

### 3. Walidacja

Zarówno dla **modelu A** (klasyfikacja pozy) jak i dla **modelu B** (klasyfikacja zdarzenia) zostały przetestowane 3 modele klasycznego uczenia maszynowego: **model SVM**, **model Random Forest**, oraz **model MLP**.

1. Walidacja **modelu A** - LeaveOneGroupOut
Dla **modelu A** (klasyfikującego pozę) została przeprowadzona walidacja **LeaveOneGroupOut**, aby podzielić klatki
sekwencjami w celu uniknięcia data leakage, gdzie uzyskano następujące rezultaty:

| **Model** | **Accuracy** | **Recall** | **Lossess** |
| --- | --- | --- | --- |
| Random Forest | 0.967 | 0.61 | 0.11 |
| SVM | 0.973 | 0.601 | 0.088 |
| MLP | 0.968 | 0.595 | 0.152 |

Oraz macierze pomyłek, które prezentują się następująco:

<img width="945" height="264" alt="image" src="https://github.com/user-attachments/assets/8d49479e-4b77-4bf2-8a68-d425af753acd" />

Na podstawie uzyskanych wyników wybrano jako najlepszy **model SVM**, który został później
wykorzystany do stworzenia pełnego pipeline’u systemu detekcji upadków.

2. Walidacja **modelu B** - LeaveOneOut
Dla **modelu B** (klasyfikującego zdarzenie) została przeprowadzona walidacja LeaveOneOut, ponieważ dane zostały już wcześniej podzielone sekwencjami na pojedyncze wektory dla każdej sekwencji, uzyskano następujące rezultaty:

| **Model** | **Accuracy** | **Recall** | **Lossess** |
| --- | --- | --- | --- |
| Random Forest | 0.933 | 0.644 | 0.32 |
| SVM | 0.844 | 0.6 | 0.325 |
| MLP | 0.844 | 0.578| 0.333 |

Oraz macierze pomyłek, które prezentują się następująco:

<img width="944" height="273" alt="image" src="https://github.com/user-attachments/assets/09a79dea-c014-4269-b7ff-ad90a551ed25" />

Na podstawie uzyskanych wyników wybrano jako najlepszy **model Random Forest**, który
został później wykorzystany do stworzenia pełnego pipeline’u systemu detekcji upadków.

### 4. Pełen system do detekcji upadków 

Pipeline detekcji upadków został zaprojektowany w następujący sposób:
1. Badany film należy przepuścić przed detektor aby uzyskać plik JSON z danymi
2. Uzyskany plik JSON umieszczamy w **systemie detekcji upadków**
3. Wyciągamy informacje o keypointach w poszczególnych klatkach z pliku JSON
4. Algorytm przechodzi po keypointach z każdej klatki, klasyfikując je Modelem A:
   
    **Jeżeli**:
   
   - Wszystkie klatki zostały sklasyfikowane jako 0 (postać nie znalazła się w
pozycji leżącej) to system KOŃCZY działanie, zwracając, że film reprezentuje
sekwencje **ADL**

   - Co najmniej 3 klatki z rzędu zostały sklasyfikowane jako 1 (postać znalazła się
w pozycji leżącej) przechodzimy do punktu 5.

6. Obliczamy wszystkie statystyki (opisane w Modelowaniu sekwencji) dla wideo z
keypointów aby uzyskać reprezentacje całego wideo w jednym tensorze
7. Uzyskany tensor zostaje sklasyfikowany przez Model B:
   
    **Jeżeli**:
   
    - Model zwróci 0 to KONIEC, zwracając, że film reprezentuje sekwencje **ADL**
    - Model zwróci 1 to KONIEC, zwracając, że film reprezentuje sekwencje **FALL**

Uzyskane wyniki zaprezentowano poniżej:

| **Model** | **Accuracy** | **Precision** |
| --- | --- | --- |
| Model A (SVM) | 0.967 | 0.644 |
| Model B (Random Forest) | 1.000 | 1.00 |

Oraz macierze pomyłek:

<img width="944" height="402" alt="image" src="https://github.com/user-attachments/assets/79291696-915c-4913-befe-e3300b3b513d" />

Wyniki są bardzo dobre, w przypadku modelu B do klasyfikacji zdarzenia, wynik 100% poprawności jest prawdopodobnie lekko mylnym wynikiem, wynik taki jest ponieważ do treningu i testowania jest mało danych (zaledwie 45 sekwencji). 

## 📈 Podsumowanie

System został przetestowany najpierw na pełnym zbiorze UR Fall Detection z następującymi rezultatami:

<img width="638" height="253" alt="image" src="https://github.com/user-attachments/assets/6705899f-aa03-4a5b-b3de-fbe8513fe332" />

System został też przetestowany na kilku sekwencjach z innego datasetu pozyskanego z strony [Kaggle](https://www.kaggle.com/datasets/payutch/fall-video-dataset):

<img width="638" height="205" alt="image" src="https://github.com/user-attachments/assets/f40174cd-c5d5-4ad0-a114-49a6c24ca9a4" />

Zatem zaprojektowany **system detekcji upadku**, oparty na dwuetapowej klasyfikacji, z modelami wytrenowanymi na zbiorze UR Fall Detection, radzi sobie bardzo dobrze, myli się w sytuacjach nie jednoznacznych gdy osoba na wideo siada lub kładzie się w bardzo dynamiczny sposób, system rozpoznaje to wtedy jako upadek, mimo iż to czynność codzienna.

<a id="env"></a>
## 🌐 Ustawienie środowiska

### 1. Klonowanie repozytorium:

Aby sklonować repozytorium:

`git clone https://github.com/KacperM33/Projekt_AIPO`

Przejść do projektu:

`cd Projekt_AIPO/`

---

### 2. Utworzenie nowego środowiska conda:

Od razu zostają zainstalowane niemalże wszystkie potrzebne pakiety w tym pytorch i torchvision.

`conda env create -f environment.yml -n nazwa_srodowiska`

> Może to chwilę potrwać.

---

### 3. Aktywacja środowiska conda:

`conda activate nazwa_srodowiska`

---

### 4. Instalacja lokalnego pakietu mmpose:

Przejść do folderu mmpose:

`cd mmpose`

I zainstalować za pomocą:

`pip install -v -e .`

---
### 5. Instalacja mmcv

Wrócić do głównego folderu projektu:

`cd..`

Zależnie od systemu podjąć dalsze kroki...

### Instalacja mmcv **(Windows)**

Na systemie Windows, można zainstalować mmcv z pliku .whl znajdującego się w folderze `mmcv_wheel`. Aby to zrobić należy użyć polecenia:

`pip install ../mmcv_wheel/mmcv-2.1.0-cp310-cp310-win_amd64.whl`

### Instalacja mmcv **(Linux)**

Na systemie Linux, można zainstalować mmcv z pliku .whl znajdującego się w folderze `mmcv_wheel`. Aby to zrobić należy użyć polecenia:

`pip install ../mmcv_wheel/mmcv-2.1.0-cp310-cp310-linux_x86_64.whl`

### Alternatywa gdyby nadal nie działało

Gdyby powyższe sposoby nie działały, można zainstalować mmcv bezpośrednio z repozytorium w tym celu:

1. `git clone -b v2.1.0 https://github.com/open-mmlab/mmcv.git`


2. `cd mmcv`


3. Windows: `set DISTUTILS_USE_SDK=1`

    Linux: `export CUDA_HOME=/usr/local/cuda-12.1`


4. `pip install -e .`

---

### 6. Instalacja kompatybilnego numpy:

`pip install numpy==1.24.4`

> Z nowszą wersją numpy skrypt nie uruchamia się.

> Errory po instalacji numpy==1.24.4 są nieistnotne (chyba, w każdym razie działa).

---
