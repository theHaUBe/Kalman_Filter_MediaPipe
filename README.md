# Filtr Kalmana z MediaPipe – Predykcja i Stabilizacja Położenia Dłoni

W tym projekcie wykonałem detekcję dłoni przy użyciu biblioteki **MediaPipe** i dodałem do niej **Filtr Kalmana** z OpenCV. Dzięki temu możliwa jest stabilna predykcja położenia punktów dłoni nawet wtedy, gdy ręka chwilowo znika z pola widzenia kamery. Dodatkowo projekt umożliwia rysowanie po wideo przy użyciu zaciśniętego kciuka i palca wskazującego.

## Funkcjonalności

1. **Detekcja dłoni z MediaPipe**
   - Wykorzystanie modułu `Hand Landmarker`.
   - Obsługa wielu landmarków dłoni (21 punktów).
   - Konwersja obrazu z kamery do formatu RGB i przetwarzanie w czasie rzeczywistym.

2. **Filtr Kalmana**
   - Implementacja klasy `Kalman_Filtering` dla 21 punktów dłoni.
   - Rekurencyjna predykcja położenia punktów.
   - Stabilizacja położenia, nawet gdy dłoń częściowo znika z obrazu.
   - Uwzględnianie dynamicznego kroku czasowego między klatkami (dt).

3. **Rysowanie po wideo**
   - Wykrywanie gestu zaciśniętego kciuka i palca wskazującego.
   - Rysowanie ścieżki punktów na wideo w czasie rzeczywistym.

## Przykład działania
