# magnetic-scanner
Testing simulations

## Setup e Utilizzo del Simulatore di Campi Magnetici 3D

## Requisiti

Per utilizzare lo script è necessario avere installato Python 3.x e i seguenti package:

```bash
pip install numpy matplotlib scipy
```

Per l'accelerazione GPU (opzionale ma consigliata):

```bash
pip install cupy numba
```

## Funzionalità Principali

Lo script implementa le seguenti funzionalità chiave:

1. **Simulazione di campi magnetici in 3D con accelerazione GPU** se disponibile, altrimenti utilizzo della CPU
2. **Ottimizzazione delle frequenze e delle ampiezze** per creare pattern di interferenza costruttiva e distruttiva nelle aree target
3. **Posizionamento ottimale dei sensori** per il rilevamento del campo magnetico risultante
4. **Posizionamento ottimale dei magneti** per ottenere un campo desiderato in un'area specifica
5. **Visualizzazione 2D e 3D** dei campi magnetici risultanti

## Esempio di Utilizzo Base

```python
from campo_magnetico_3d import SimulatoreCampoMagnetico3D

# Inizializza il simulatore
simulatore = SimulatoreCampoMagnetico3D(dimensioni_spazio=(100, 100, 100), risoluzione=2.0)

# Aggiungi magneti
simulatore.aggiungi_magnete(posizione=(30, 50, 50), momento_magnetico=(0, 0, 5), frequenza=100)
simulatore.aggiungi_magnete(posizione=(70, 50, 50), momento_magnetico=(0, 0, -5), frequenza=100)

# Calcola il campo magnetico
simulatore.calcola_campo_totale(tempo=0.0)

# Visualizza il campo
simulatore.visualizza_campo(piano='xy', posizione=50)
```

## Ottimizzazione delle Frequenze

```python
# Definisci un'area target (per esempio un'area corrispondente a un corpo in un esame medico)
area_target = ((40, 60), (40, 60), (40, 60))  # ((x_min, x_max), (y_min, y_max), (z_min, z_max))

# Ottimizza le frequenze per l'interferenza costruttiva nell'area target
frequenze_ottimali = simulatore.ottimizza_frequenze(
    area_target, 
    range_frequenze=(50, 500),
    metodo='interferenza_costruttiva'
)

# Ricalcola il campo con le frequenze ottimizzate
simulatore.calcola_campo_totale(tempo=0.0)
```

## Ottimizzazione delle Ampiezze

```python
# Ottimizza le ampiezze dei magneti per massimizzare l'intensità e uniformità del campo nell'area target
ampiezze_ottimali = simulatore.ottimizza_ampiezze(area_target)

# Ricalcola il campo con le ampiezze ottimizzate
simulatore.calcola_campo_totale(tempo=0.0)
```

## Posizionamento Ottimale dei Sensori

```python
# Trova le posizioni ottimali per posizionare 3 sensori
posizioni_sensori = simulatore.ottimizza_posizioni_sensori(num_sensori=3)

# Visualizza il campo con i sensori
simulatore.visualizza_campo_3d(soglia=0.3)
```

## Posizionamento Ottimale dei Magneti

```python
# Trova le posizioni ottimali per posizionare 4 magneti intorno all'area target
posizioni_magneti = simulatore.ottimizza_posizioni_magneti(
    num_magneti=4, 
    area_target=area_target
)

# Visualizza il campo con i magneti ottimizzati
simulatore.visualizza_campo_3d(soglia=0.3)
```

## Considerazioni Fisiche

Il simulatore incorpora diversi principi fisici importanti:

1. **Effetti Relativistici**: Include il ritardo di propagazione del campo in base alla velocità della luce
2. **Proprietà dell'Aria**: Considera la permeabilità magnetica leggermente diversa dell'aria rispetto al vuoto
3. **Interferenza di Campo**: Simula l'interferenza costruttiva e distruttiva tra campi magnetici oscillanti
4. **Attenuazione del Campo**: Modella l'attenuazione del campo magnetico in funzione della distanza

## Note sull'Utilizzo per Applicazioni Mediche

Per applicazioni mediche (come la risonanza magnetica), è importante:

1. Specificare un'area target che rappresenti l'area del corpo da esaminare
2. Utilizzare frequenze nel range appropriato per la risonanza dei nuclei di interesse
3. Posizionare i sensori in modo ottimale per rilevare le variazioni di campo dovute all'interazione con i tessuti
4. Considerare che i tessuti biologici hanno proprietà magnetiche diverse dall'aria

## Visualizzazione Avanzata

```python
# Visualizza il campo in un piano specifico
simulatore.visualizza_campo(
    piano='xy',         # Piano di visualizzazione (xy, xz, yz)
    posizione=50,       # Posizione del piano lungo l'asse perpendicolare
    tempo=0.0,          # Tempo per campi variabili
    magnitudine=True,   # Visualizza la magnitudine del campo
    componente=None     # Oppure specifica una componente (0=x, 1=y, 2=z)
)

# Visualizzazione 3D del campo
simulatore.visualizza_campo_3d(
    tempo=0.0,   # Tempo per la visualizzazione
    soglia=0.3   # Soglia per filtrare punti con campo debole (0-1)
)
```

## Limitazioni e Considerazioni

1. La risoluzione della simulazione influisce significativamente sulle prestazioni
2. L'ottimizzazione di frequenze, ampiezze e posizioni richiede tempo e risorse computazionali
3. Per simulazioni di alta precisione, considerare l'uso di un sistema con GPU potente
4. Per applicazioni reali, calibrare il modello con dati sperimentali