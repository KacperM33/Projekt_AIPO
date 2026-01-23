## 1. Ustawienie środowiska

### 1. Kluczowe wymagania

- Python 3.10.18
- PyTorch 2.5.1
- mmcv 2.1.0
- cuda 12.1

### 2. Klonowanie repozytorium:


Aby sklonować repozytorium:

`git clone https://github.com/KacperM33/Projekt_AIPO`

Przejść do projektu:

`cd Projekt_AIPO/`

---

### 3. Utworzenie nowego środowiska conda:

Od razu zostają zainstalowane niemalże wszystkie potrzebne pakiety w tym pytorch i torchvision.

`conda env create -f environment.yml -n nazwa_srodowiska`

> Może to chwilę potrwać.

---

### 4. Aktywacja środowiska conda:

`conda activate nazwa_srodowiska`

---

### 5. Instalacja lokalnego pakietu mmpose:

Przejść do folderu mmpose:

`cd mmpose`

I zainstalować za pomocą:

`pip install -v -e .`

---
### 6. Instalacja mmcv

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

### 7. Instalacja kompatybilnego numpy:

`pip install numpy==1.24.4`

> Z nowszą wersją numpy skrypt nie uruchamia się.

> Errory po instalacji numpy==1.24.4 są nieistnotne (chyba, w każdym razie działa).

---

### 8. Koniec
