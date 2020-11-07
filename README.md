# Find the nose!
## KN Solvro zadanie rekrutacyjnie z kategorii ML 
* Plik **sprawozadnie.pdf** to krótki przegląd zbierania i obróbki danych, wyboru modelu i treningu. 
* Notatnik **test.ipynb** wczytuje już wytrenowany model i sprawdza jak sieć radzi sobie na kilku przykładowych zdjęciach. 
* Notatnik **model.ipynb** to budowa, trening i porównanie kilku modeli. Żeby przetestować go dla mniejszych danych wystarczy podmienić wczytywane na samym początki macierze X i Y na X_small i Y_small (zakomentowany kod z komórki pod **Load data**).
**Uwaga:** dla pary (X_small, Y_small) trzeba zmniejszyć rozmiary X_val i X_test z komórki **Split data into training, validation and test sets**.
* Notatnik **data.ipynb** odpowiadał za wczytanie i preprocessing zdjęć, tutaj żeby uruchomić kod trzeba ręcznie pobrać [Facial Keyponts (68) dataset](https://www.kaggle.com/tarunkr/facial-keypoints-68-dataset), zmienić nazwę pobranego folderu na **Facial Keypoints** i przenieść go do folder data. 
