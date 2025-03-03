import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
from scipy.optimize import minimize
from scipy.constants import mu_0, c, epsilon_0

# Verificare se CUDA è disponibile e importare moduli GPU
try:
    import cupy as cp
    from numba import cuda, float64, complex128, int32
    HAS_GPU = True
    print("GPU accelerazione disponibile con CUDA")
except ImportError:
    HAS_GPU = False
    print("GPU accelerazione non disponibile, utilizzo CPU")
    cp = np  # Use numpy as a fallback

class SimulatoreCampoMagnetico3D:
    def __init__(self, dimensioni_spazio=(100, 100, 100), risoluzione=1.0):
        """
        Inizializza il simulatore di campo magnetico 3D

        Parametri:
        - dimensioni_spazio: tuple (x, y, z) dimensioni dello spazio in cm
        - risoluzione: risoluzione della griglia in cm
        """
        self.dimensioni = dimensioni_spazio
        self.risoluzione = risoluzione

        # Calcolo numero di punti nella griglia
        self.nx = int(dimensioni_spazio[0] / risoluzione) + 1
        self.ny = int(dimensioni_spazio[1] / risoluzione) + 1
        self.nz = int(dimensioni_spazio[2] / risoluzione) + 1

        # Creazione della griglia di coordinate
        x = np.linspace(0, dimensioni_spazio[0], self.nx)
        y = np.linspace(0, dimensioni_spazio[1], self.ny)
        z = np.linspace(0, dimensioni_spazio[2], self.nz)

        # Mesh grid per tutti i punti nello spazio
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

        # Inizializzazione delle variabili per magneti e sensori
        self.magneti = []
        self.sensori = []

        # Proprietà del mezzo (aria)
        self.permeabilita = mu_0  # Permeabilità magnetica del vuoto
        self.permittivita = epsilon_0  # Permittività elettrica del vuoto
        self.velocita_luce = c  # Velocità della luce nel vuoto

        # Fattore di correzione per l'aria (leggermente diverso dal vuoto)
        self.fattore_aria = 1.0000004  # Permeabilità relativa dell'aria

        # Campo magnetico risultante (inizializzato a zero)
        if HAS_GPU:
            self.campo_B = cp.zeros((self.nx, self.ny, self.nz, 3), dtype=cp.complex128)
        else:
            self.campo_B = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.complex128)

    def aggiungi_magnete(self, posizione, momento_magnetico, frequenza, fase=0.0):
        """
        Aggiunge un magnete allo spazio

        Parametri:
        - posizione: tuple (x, y, z) posizione del magnete in cm
        - momento_magnetico: tuple (mx, my, mz) momento magnetico in A·m²
        - frequenza: frequenza di oscillazione in Hz
        - fase: fase iniziale in radianti
        """
        self.magneti.append({
            'posizione': np.array(posizione),
            'momento': np.array(momento_magnetico),
            'frequenza': frequenza,
            'fase': fase,
            'ampiezza': np.linalg.norm(momento_magnetico)
        })
        print(f"Magnete aggiunto a posizione {posizione}, frequenza {frequenza} Hz")

    def aggiungi_sensore(self, posizione):
        """
        Aggiunge un sensore di campo magnetico

        Parametri:
        - posizione: tuple (x, y, z) posizione del sensore in cm
        """
        idx_x = int(posizione[0] / self.risoluzione)
        idx_y = int(posizione[1] / self.risoluzione)
        idx_z = int(posizione[2] / self.risoluzione)

        # Assicurarsi che gli indici siano all'interno dei limiti
        idx_x = max(0, min(idx_x, self.nx - 1))
        idx_y = max(0, min(idx_y, self.ny - 1))
        idx_z = max(0, min(idx_z, self.nz - 1))

        self.sensori.append({
            'posizione': np.array(posizione),
            'indici': (idx_x, idx_y, idx_z)
        })
        print(f"Sensore aggiunto a posizione {posizione}")

    def calcola_campo_magnetico_punto(self, punto, magnete, tempo):
        """
        Calcola il campo magnetico in un punto dello spazio dovuto a un singolo magnete
        considerando anche gli effetti relativistici

        Parametri:
        - punto: array [x, y, z] posizione in cui calcolare il campo
        - magnete: dizionario con informazioni sul magnete
        - tempo: tempo in secondi

        Ritorna:
        - campo_B: array [Bx, By, Bz] campo magnetico nel punto
        """
        r = punto - magnete['posizione']
        distanza = np.linalg.norm(r)

        if distanza < 1e-10:  # Evitare divisione per zero
            return np.zeros(3)

        # Vettore unitario nella direzione di r
        r_hat = r / distanza

        # Calcolo del campo magnetico statico (formula del dipolo)
        m = magnete['momento']

        # Termine dipolo (formula classica)
        B_statico = (self.permeabilita * self.fattore_aria / (4 * np.pi)) * (
            3 * r_hat * np.dot(r_hat, m) - m
        ) / (distanza**3)

        # Calcolo del campo magnetico oscillante nel tempo
        omega = 2 * np.pi * magnete['frequenza']
        fase = magnete['fase']

        # Fattore di ritardo temporale dovuto alla propagazione finita
        ritardo = distanza / self.velocita_luce

        # Calcolo del campo con ritardo di propagazione (effetto relativistico)
        tempo_ritardato = tempo - ritardo

        # Termine di oscillazione con fase
        oscillazione = np.exp(1j * (omega * tempo_ritardato + fase))

        # Campo risultante con effetti relativistici
        # Includere fattore di attenuazione 1/r per onde sferiche
        fattore_attenuazione = 1.0 / (1.0 + distanza / 100.0)  # Attenuazione semplificata

        B = B_statico * oscillazione * fattore_attenuazione

        return B

    @staticmethod
    @cuda.jit
    def _calcola_campo_gpu(X, Y, Z, pos_magneti, mom_magneti, freq, fasi, amp,
                          tempo, permeabilita, fattore_aria, vel_luce, campo_B):
        """
        Kernel CUDA per calcolare il campo magnetico in parallelo
        """
        i, j, k = cuda.grid(3)
        nx, ny, nz = campo_B.shape[0], campo_B.shape[1], campo_B.shape[2]

        if i < nx and j < ny and k < nz:
            punto = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])

            for mag_idx in range(len(pos_magneti)):
                r_x = punto[0] - pos_magneti[mag_idx][0]
                r_y = punto[1] - pos_magneti[mag_idx][1]
                r_z = punto[2] - pos_magneti[mag_idx][2]

                distanza_sq = r_x*r_x + r_y*r_y + r_z*r_z
                distanza = distanza_sq**0.5

                if distanza > 1e-10:  # Evitare divisione per zero
                    # Vettore unitario
                    r_hat_x = r_x / distanza
                    r_hat_y = r_y / distanza
                    r_hat_z = r_z / distanza

                    # Momento magnetico del magnete
                    m_x = mom_magneti[mag_idx][0]
                    m_y = mom_magneti[mag_idx][1]
                    m_z = mom_magneti[mag_idx][2]

                    # Prodotto scalare r_hat · m
                    dot_product = r_hat_x * m_x + r_hat_y * m_y + r_hat_z * m_z

                    # Calcolo del campo magnetico statico
                    prefactor = (permeabilita * fattore_aria / (4 * 3.14159)) / (distanza**3)

                    B_x = prefactor * (3 * r_hat_x * dot_product - m_x)
                    B_y = prefactor * (3 * r_hat_y * dot_product - m_y)
                    B_z = prefactor * (3 * r_hat_z * dot_product - m_z)

                    # Fattore di ritardo temporale
                    ritardo = distanza / vel_luce
                    tempo_ritardato = tempo - ritardo

                    # Oscillazione
                    omega = 2 * 3.14159 * freq[mag_idx]
                    fase = fasi[mag_idx]

                    # Calcolo di exp(i * (omega * t + fase))
                    cos_val = cuda.cos(omega * tempo_ritardato + fase)
                    sin_val = cuda.sin(omega * tempo_ritardato + fase)

                    # Fattore di attenuazione
                    fattore_attenuazione = 1.0 / (1.0 + distanza / 100.0)

                    # Aggiunta al campo totale (parte reale e immaginaria)
                    campo_B[i, j, k, 0] += complex(B_x * cos_val * fattore_attenuazione,
                                                 B_x * sin_val * fattore_attenuazione)
                    campo_B[i, j, k, 1] += complex(B_y * cos_val * fattore_attenuazione,
                                                 B_y * sin_val * fattore_attenuazione)
                    campo_B[i, j, k, 2] += complex(B_z * cos_val * fattore_attenuazione,
                                                 B_z * sin_val * fattore_attenuazione)

    def calcola_campo_totale(self, tempo=0.0):
        """
        Calcola il campo magnetico totale in tutto lo spazio al tempo specificato
        utilizzando GPU se disponibile

        Parametri:
        - tempo: tempo in secondi per calcolare il campo
        """
        t_inizio = time.time()

        # Azzerare il campo totale
        if HAS_GPU:
            self.campo_B = cp.zeros((self.nx, self.ny, self.nz, 3), dtype=cp.complex128)
        else:
            self.campo_B = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.complex128)

        if not self.magneti:
            print("Nessun magnete presente. Aggiungi magneti prima di calcolare il campo.")
            return

        if HAS_GPU and len(self.magneti) > 0:
            # Preparazione dati per GPU
            pos_magneti = np.array([m['posizione'] for m in self.magneti])
            mom_magneti = np.array([m['momento'] for m in self.magneti])
            frequenze = np.array([m['frequenza'] for m in self.magneti])
            fasi = np.array([m['fase'] for m in self.magneti])
            ampiezze = np.array([m['ampiezza'] for m in self.magneti])

            # Trasferimento dati alla GPU
            d_X = cp.asarray(self.X)
            d_Y = cp.asarray(self.Y)
            d_Z = cp.asarray(self.Z)
            d_pos_magneti = cp.asarray(pos_magneti)
            d_mom_magneti = cp.asarray(mom_magneti)
            d_freq = cp.asarray(frequenze)
            d_fasi = cp.asarray(fasi)
            d_amp = cp.asarray(ampiezze)

            # Configurazione dei blocchi CUDA
            threads_per_block = (8, 8, 8)
            blocks_per_grid_x = (self.nx + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (self.ny + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid_z = (self.nz + threads_per_block[2] - 1) // threads_per_block[2]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

            # Esecuzione del kernel CUDA
            self._calcola_campo_gpu[blocks_per_grid, threads_per_block](
                d_X, d_Y, d_Z, d_pos_magneti, d_mom_magneti, d_freq, d_fasi, d_amp,
                tempo, self.permeabilita, self.fattore_aria, self.velocita_luce,
                self.campo_B
            )
            cuda.synchronize()

        else:
            # Versione CPU
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        punto = np.array([self.X[i, j, k], self.Y[i, j, k], self.Z[i, j, k]])

                        # Somma dei contributi di ogni magnete
                        for magnete in self.magneti:
                            self.campo_B[i, j, k] += self.calcola_campo_magnetico_punto(punto, magnete, tempo)

        t_fine = time.time()
        print(f"Calcolo del campo magnetico completato in {t_fine - t_inizio:.3f} secondi")

        return self.campo_B
    
    def ottimizza_posizioni_magneti(self, num_magneti, area_target, area_possibile=None):
        """
        Determina le posizioni ottimali dove posizionare i magneti per ottenere
        un campo magnetico ottimale nell'area target

        Parametri:
        - num_magneti: numero di magneti da posizionare
        - area_target: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) area di interesse
        - area_possibile: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) area dove possono essere posizionati i magneti

        Ritorna:
        - posizioni_ottimali: lista di posizioni ottimali per i magneti
        """
        # Se non specificata, usa tutto lo spazio disponibile eccetto l'area target
        if area_possibile is None:
            area_possibile = ((0, self.dimensioni[0]), (0, self.dimensioni[1]), (0, self.dimensioni[2]))

        # Momento magnetico e frequenza predefiniti
        momento_predefinito = np.array([0.0, 0.0, 1.0])  # Orientato lungo z
        frequenza_predefinita = 100.0  # Hz

        # Indici dell'area target nella griglia
        x_min_idx = max(0, int(area_target[0][0] / self.risoluzione))
        x_max_idx = min(self.nx - 1, int(area_target[0][1] / self.risoluzione))
        y_min_idx = max(0, int(area_target[1][0] / self.risoluzione))
        y_max_idx = min(self.ny - 1, int(area_target[1][1] / self.risoluzione))
        z_min_idx = max(0, int(area_target[2][0] / self.risoluzione))
        z_max_idx = min(self.nz - 1, int(area_target[2][1] / self.risoluzione))
        

        # Funzione per generare posizioni candidate casuali nell'area possibile
        def genera_posizioni_casuali():
            pos_x = np.random.uniform(area_possibile[0][0], area_possibile[0][1], num_magneti)
            pos_y = np.random.uniform(area_possibile[1][0], area_possibile[1][1], num_magneti)
            pos_z = np.random.uniform(area_possibile[2][0], area_possibile[2][1], num_magneti)
            return np.column_stack((pos_x, pos_y, pos_z))

        def funzione_obiettivo(posizioni_1d):
            """
            Funzione da minimizzare per ottimizzare le posizioni dei magneti
            """
            # Riorganizza l'array 1D in array 2D di posizioni [num_magneti, 3]
            posizioni = posizioni_1d.reshape(num_magneti, 3)

            # Salva i magneti attuali
            magneti_precedenti = self.magneti.copy()
            self.magneti = []

            # Crea magneti con le nuove posizioni
            for pos in posizioni:
                self.aggiungi_magnete(pos, momento_predefinito, frequenza_predefinita)

            # Calcola il campo magnetico con le nuove posizioni
            self.calcola_campo_totale()

            # Estrai l'ampiezza del campo nell'area target
            campo_area = np.abs(self.campo_B[x_min_idx:x_max_idx+1,
                                          y_min_idx:y_max_idx+1,
                                          z_min_idx:z_max_idx+1])

            # Calcola metriche di ottimizzazione
            intensita_media = np.mean(np.linalg.norm(campo_area, axis=3))
            uniformita = np.std(np.linalg.norm(campo_area, axis=3)) / intensita_media if intensita_media > 0 else np.inf

            # Obiettivo: massima intensità con massima uniformità nell'area target
            metrica = uniformita / intensita_media if intensita_media > 0 else np.inf

            # Ripristina i magneti precedenti
            self.magneti = magneti_precedenti

            return metrica

        # Genera posizioni iniziali casuali
        posizioni_iniziali = genera_posizioni_casuali().flatten()

        # Esegui l'ottimizzazione
        print("Ottimizzazione delle posizioni dei magneti in corso...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risultato = minimize(
                funzione_obiettivo,
                posizioni_iniziali,
                method='Nelder-Mead',
                options={'maxiter': 100}
            )

        # Estrai le posizioni ottimali
        posizioni_ottimali = risultato.x.reshape(num_magneti, 3)

        print("Posizioni ottimali dei magneti:")
        for i, pos in enumerate(posizioni_ottimali):
            print(f"Magnete {i}: posizione = {pos}")

        return posizioni_ottimali

    def ottimizza_frequenze(self, area_target, range_frequenze=(1, 1000),
                           num_frequenze=5, metodo='interferenza_costruttiva'):
        """
        Ottimizza le frequenze dei magneti per ottenere un'interferenza ottimale
        nell'area target

        Parametri:
        - area_target: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) area di interesse
        - range_frequenze: tuple (min, max) range di frequenze in Hz
        - num_frequenze: numero di frequenze da ottimizzare
        - metodo: 'interferenza_costruttiva' o 'interferenza_distruttiva'

        Ritorna:
        - frequenze_ottimali: lista di frequenze ottimali per i magneti
        """
        if len(self.magneti) == 0:
            print("Nessun magnete presente. Aggiungi magneti prima di ottimizzare le frequenze.")
            return []

        # Inizializzazione con frequenze casuali nel range
        frequenze_iniziali = np.random.uniform(
            range_frequenze[0], range_frequenze[1], min(len(self.magneti), num_frequenze)
        )

        # Indici dell'area target nella griglia
        x_min_idx = max(0, int(area_target[0][0] / self.risoluzione))
        x_max_idx = min(self.nx - 1, int(area_target[0][1] / self.risoluzione))
        y_min_idx = max(0, int(area_target[1][0] / self.risoluzione))
        y_max_idx = min(self.ny - 1, int(area_target[1][1] / self.risoluzione))
        z_min_idx = max(0, int(area_target[2][0] / self.risoluzione))
        z_max_idx = min(self.nz - 1, int(area_target[2][1] / self.risoluzione))

        def funzione_obiettivo(frequenze):
            """
            Funzione da minimizzare/massimizzare per ottimizzare le frequenze
            """
            # Aggiorna le frequenze dei magneti
            for i, freq in enumerate(frequenze):
                if i < len(self.magneti):
                    self.magneti[i]['frequenza'] = freq

            # Calcola il campo magnetico con le nuove frequenze
            self.calcola_campo_totale(tempo=0.0)

            # Estrai l'ampiezza del campo nell'area target
            campo_area = np.abs(self.campo_B[x_min_idx:x_max_idx+1,
                                            y_min_idx:y_max_idx+1,
                                            z_min_idx:z_max_idx+1])

            # Calcola la metrica in base al metodo
            if metodo == 'interferenza_costruttiva':
                # Massimizzare l'intensità media nell'area target
                metrica = -np.mean(np.linalg.norm(campo_area, axis=3))
            else:  # interferenza_distruttiva
                # Minimizzare l'intensità media nell'area target
                metrica = np.mean(np.linalg.norm(campo_area, axis=3))

            return metrica

        # Esegui l'ottimizzazione
        print(f"Ottimizzazione delle frequenze in corso con il metodo '{metodo}'...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risultato = minimize(
                funzione_obiettivo,
                frequenze_iniziali,
                method='Nelder-Mead',
                bounds=[(range_frequenze[0], range_frequenze[1])] * len(frequenze_iniziali),
                options={'maxiter': 100}
            )

        frequenze_ottimali = risultato.x

        # Aggiorna le frequenze dei magneti con i valori ottimali
        for i, freq in enumerate(frequenze_ottimali):
            if i < len(self.magneti):
                self.magneti[i]['frequenza'] = freq
                print(f"Magnete {i}: frequenza ottimale = {freq:.2f} Hz")

        return frequenze_ottimali
    
    def ottimizza_ampiezze(self, area_target, range_ampiezze=(0.1, 10.0)):
        """
        Ottimizza le ampiezze dei magneti per ottenere un campo ottimale nell'area target

        Parametri:
        - area_target: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) area di interesse
        - range_ampiezze: tuple (min, max) range di ampiezze relative

        Ritorna:
        - ampiezze_ottimali: lista di fattori di scala per le ampiezze dei magneti
        """
        if len(self.magneti) == 0:
            print("Nessun magnete presente. Aggiungi magneti prima di ottimizzare le ampiezze.")
            return []

        # Inizializzazione con ampiezze unitarie
        ampiezze_iniziali = np.ones(len(self.magneti))

        # Salva i momenti magnetici originali
        momenti_originali = [np.copy(m['momento']) for m in self.magneti]
        ampiezze_originali = [np.linalg.norm(m['momento']) for m in self.magneti]

        # Indici dell'area target nella griglia
        x_min_idx = max(0, int(area_target[0][0] / self.risoluzione))
        x_max_idx = min(self.nx - 1, int(area_target[0][1] / self.risoluzione))
        y_min_idx = max(0, int(area_target[1][0] / self.risoluzione))
        y_max_idx = min(self.ny - 1, int(area_target[1][1] / self.risoluzione))
        z_min_idx = max(0, int(area_target[2][0] / self.risoluzione))
        z_max_idx = min(self.nz - 1, int(area_target[2][1] / self.risoluzione))

        def funzione_obiettivo(fattori_scala):
            """
            Funzione da minimizzare per ottimizzare le ampiezze
            """
            # Aggiorna i momenti magnetici con i fattori di scala
            for i, fattore in enumerate(fattori_scala):
                if i < len(self.magneti):
                    direzione = momenti_originali[i] / ampiezze_originali[i]
                    self.magneti[i]['momento'] = direzione * (ampiezze_originali[i] * fattore)
                    self.magneti[i]['ampiezza'] = ampiezze_originali[i] * fattore

            # Calcola il campo magnetico con le nuove ampiezze
            self.calcola_campo_totale(tempo=0.0)

            # Estrai l'ampiezza del campo nell'area target
            campo_area = np.abs(self.campo_B[x_min_idx:x_max_idx+1,
                                            y_min_idx:y_max_idx+1,
                                            z_min_idx:z_max_idx+1])

            # Massimizzare l'uniformità e l'intensità nell'area target
            intensita_media = np.mean(np.linalg.norm(campo_area, axis=3))
            variazione = np.std(np.linalg.norm(campo_area, axis=3)) / intensita_media if intensita_media > 0 else np.inf

            # Obiettivo: massima intensità con minima variazione
            metrica = variazione / intensita_media if intensita_media > 0 else np.inf

            return metrica

        # Esegui l'ottimizzazione
        print("Ottimizzazione delle ampiezze in corso...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risultato = minimize(
                funzione_obiettivo,
                ampiezze_iniziali,
                method='Nelder-Mead',
                bounds=[(range_ampiezze[0], range_ampiezze[1])] * len(ampiezze_iniziali),
                options={'maxiter': 100}
            )

        ampiezze_ottimali = risultato.x

        # Aggiorna i momenti magnetici con i valori ottimali
        for i, fattore in enumerate(ampiezze_ottimali):
            if i < len(self.magneti):
                direzione = momenti_originali[i] / ampiezze_originali[i]
                self.magneti[i]['momento'] = direzione * (ampiezze_originali[i] * fattore)
                self.magneti[i]['ampiezza'] = ampiezze_originali[i] * fattore
                print(f"Magnete {i}: ampiezza ottimale = {self.magneti[i]['ampiezza']:.2f} A·m²")

        return ampiezze_ottimali
    
    def ottimizza_posizioni_sensori(self, num_sensori, area_possibile=None):
        """
        Determina le posizioni ottimali dove posizionare i sensori per rilevare
        il campo magnetico risultante

        Parametri:
        - num_sensori: numero di sensori da posizionare
        - area_possibile: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) area dove possono essere posizionati i sensori

        Ritorna:
        - posizioni_sensori: lista di posizioni ottimali per i sensori
        """
        if not self.magneti:
            print("Nessun magnete presente. Aggiungi magneti prima di ottimizzare le posizioni dei sensori.")
            return []

        # Se non specificata, usa tutto lo spazio disponibile
        if area_possibile is None:
            area_possibile = ((0, self.dimensioni[0]), (0, self.dimensioni[1]), (0, self.dimensioni[2]))

        # Calcola il campo magnetico attuale
        self.calcola_campo_totale()

        # Converti l'array del campo da complesso a reale (ampiezza)
        campo_ampiezza = np.linalg.norm(np.abs(self.campo_B), axis=3)

        # Trova i punti di massimo gradiente del campo
        # Calcola il gradiente del campo in ogni direzione
        grad_x = np.gradient(campo_ampiezza, axis=0)
        grad_y = np.gradient(campo_ampiezza, axis=1)
        grad_z = np.gradient(campo_ampiezza, axis=2)

        # Calcola la magnitudine del gradiente
        gradiente_totale = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

        # Filtra i punti nell'area possibile
        x_min_idx = max(0, int(area_possibile[0][0] / self.risoluzione))
        x_max_idx = min(self.nx - 1, int(area_possibile[0][1] / self.risoluzione))
        y_min_idx = max(0, int(area_possibile[1][0] / self.risoluzione))
        y_max_idx = min(self.ny - 1, int(area_possibile[1][1] / self.risoluzione))
        z_min_idx = max(0, int(area_possibile[2][0] / self.risoluzione))
        z_max_idx = min(self.nz - 1, int(area_possibile[2][1] / self.risoluzione))

        gradiente_area = gradiente_totale[x_min_idx:x_max_idx+1,
                                          y_min_idx:y_max_idx+1,
                                          z_min_idx:z_max_idx+1]

        # Appiattisci l'array per trovare i massimi
        gradiente_flat = gradiente_area.flatten()
        indici_massimi = np.argsort(gradiente_flat)[-num_sensori:]

        # Converti indici appiattiti in coordinate 3D
        shape_area = gradiente_area.shape
        posizioni_sensori = []

        for idx_flat in indici_massimi:
            # Converti indice flat in indici 3D nell'area
            idx_z_area = idx_flat % shape_area[2]
            idx_y_area = (idx_flat // shape_area[2]) % shape_area[1]
            idx_x_area = idx_flat // (shape_area[1] * shape_area[2])

            # Converti in indici globali
            idx_x = idx_x_area + x_min_idx
            idx_y = idx_y_area + y_min_idx
            idx_z = idx_z_area + z_min_idx

            # Converti in coordinata fisica
            x = idx_x * self.risoluzione
            y = idx_y * self.risoluzione
            z = idx_z * self.risoluzione

            posizioni_sensori.append(np.array([x, y, z]))

        # Aggiungi i sensori
        self.sensori = []
        for pos in posizioni_sensori:
            self.aggiungi_sensore(pos)

        return posizioni_sensori

    def visualizza_campo(self, piano='xy', posizione=None, tempo=0.0,
                        magnitudine=True, componente=None):
        """
        Visualizza il campo magnetico in un piano specifico

        Parametri:
        - piano: 'xy', 'xz', o 'yz' per specificare il piano
        - posizione: posizione del piano lungo l'asse perpendicolare
        - tempo: tempo in secondi per la visualizzazione
        - magnitudine: se True, visualizza la magnitudine del campo
        - componente: None, 0, 1, o 2 per visualizzare componenti specifiche (x, y, z)
        """
        if self.campo_B is None or np.all(self.campo_B == 0):
            self.calcola_campo_totale(tempo)

        # Converti a NumPy se è un array CuPy
        if HAS_GPU and isinstance(self.campo_B, cp.ndarray):
            campo = cp.asnumpy(self.campo_B)
        else:
            campo = self.campo_B

        # Determina gli indici per il piano di visualizzazione
        if posizione is None:
            if piano == 'xy':
                posizione = self.dimensioni[2] / 2
            elif piano == 'xz':
                posizione = self.dimensioni[1] / 2
            else:  # yz
                posizione = self.dimensioni[0] / 2

        # Converti posizione in indice
        if piano == 'xy':
            idx = min(self.nz - 1, max(0, int(posizione / self.risoluzione)))
            campo_slice = campo[:, :, idx]
            extent = [0, self.dimensioni[0], 0, self.dimensioni[1]]
            xlabel, ylabel = 'X (cm)', 'Y (cm)'
        elif piano == 'xz':
            idx = min(self.ny - 1, max(0, int(posizione / self.risoluzione)))
            campo_slice = campo[:, idx, :]
            extent = [0, self.dimensioni[0], 0, self.dimensioni[2]]
            xlabel, ylabel = 'X (cm)', 'Z (cm)'
        else:  # yz
            idx = min(self.nx - 1, max(0, int(posizione / self.risoluzione)))
            campo_slice = campo[idx, :, :]
            extent = [0, self.dimensioni[1], 0, self.dimensioni[2]]
            xlabel, ylabel = 'Y (cm)', 'Z (cm)'

        # Estrai magnitudine o componente specifica
        if magnitudine:
            campo_viz = np.linalg.norm(np.abs(campo_slice), axis=2)
            title = f'Magnitudine del campo magnetico nel piano {piano.upper()}, z={posizione} cm'
        elif componente is not None:
            campo_viz = np.abs(campo_slice[:, :, componente])
            componenti = ['X', 'Y', 'Z']
            title = f'Componente {componenti[componente]} del campo magnetico nel piano {piano.upper()}, z={posizione} cm'
        else:
            campo_viz = np.linalg.norm(np.abs(campo_slice), axis=2)
            title = f'Magnitudine del campo magnetico nel piano {piano.upper()}, z={posizione} cm'

        # Crea la figura
        plt.figure(figsize=(10, 8))
        im = plt.imshow(campo_viz.T, origin='lower', extent=extent, cmap='viridis',
                        aspect='equal', interpolation='bilinear')
        plt.colorbar(im, label='Intensità del campo (T)')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Posizioni dei magneti e sensori
        if piano == 'xy':
            for i, magnete in enumerate(self.magneti):
                pos = magnete['posizione']
                if abs(pos[2] - posizione) < 5:  # Magneti vicino al piano
                    plt.plot(pos[0], pos[1], 'ro', markersize=10, label=f'Magnete {i+1}' if i == 0 else "")

            for i, sensore in enumerate(self.sensori):
                pos = sensore['posizione']
                if abs(pos[2] - posizione) < 5:  # Sensori vicino al piano
                    plt.plot(pos[0], pos[1], 'bs', markersize=8, label=f'Sensore {i+1}' if i == 0 else "")

        elif piano == 'xz':
            for i, magnete in enumerate(self.magneti):
                pos = magnete['posizione']
                if abs(pos[1] - posizione) < 5:  # Magneti vicino al piano
                    plt.plot(pos[0], pos[2], 'ro', markersize=10, label=f'Magnete {i+1}' if i == 0 else "")

            for i, sensore in enumerate(self.sensori):
                pos = sensore['posizione']
                if abs(pos[1] - posizione) < 5:  # Sensori vicino al piano
                    plt.plot(pos[0], pos[2], 'bs', markersize=8, label=f'Sensore {i+1}' if i == 0 else "")

        else:  # yz
            for i, magnete in enumerate(self.magneti):
                pos = magnete['posizione']
                if abs(pos[0] - posizione) < 5:  # Magneti vicino al piano
                    plt.plot(pos[1], pos[2], 'ro', markersize=10, label=f'Magnete {i+1}' if i == 0 else "")

            for i, sensore in enumerate(self.sensori):
                pos = sensore['posizione']
                if abs(pos[0] - posizione) < 5:  # Sensori vicino al piano
                    plt.plot(pos[1], pos[2], 'bs', markersize=8, label=f'Sensore {i+1}' if i == 0 else "")

        # Aggiungi legenda se ci sono elementi
        if self.magneti or self.sensori:
            plt.legend()

        plt.tight_layout()
        plt.show()
    
    def visualizza_campo_3d(self, tempo=0.0, soglia=0.5):
        """
        Visualizza il campo magnetico in 3D

        Parametri:
        - tempo: tempo in secondi per la visualizzazione
        - soglia: valore di soglia per filtrare punti con campo debole (0-1 come percentuale del massimo)
        """
        if self.campo_B is None or np.all(self.campo_B == 0):
            self.calcola_campo_totale(tempo)

        # Converti a NumPy se è un array CuPy
        if HAS_GPU and isinstance(self.campo_B, cp.ndarray):
            campo = cp.asnumpy(self.campo_B)
        else:
            campo = self.campo_B

        # Calcola la magnitudine del campo in ogni punto
        magnitudine = np.linalg.norm(np.abs(campo), axis=3)

        # Normalizza la magnitudine e applica la soglia
        max_mag = np.max(magnitudine)
        if max_mag > 0:
            magnitudine_norm = magnitudine / max_mag
            mask = magnitudine_norm > soglia
        else:
            mask = np.ones_like(magnitudine, dtype=bool)

        # Crea indici per i punti sopra la soglia
        indices = np.where(mask)

        # Se non ci sono punti sopra la soglia, riduci la soglia
        if len(indices[0]) == 0:
            print("Nessun punto sopra la soglia. Riduzione automatica della soglia...")
            soglia = 0.1
            mask = magnitudine_norm > soglia
            indices = np.where(mask)

        # Estrai le coordinate e i valori del campo per i punti sopra la soglia
        x = self.X[indices]
        y = self.Y[indices]
        z = self.Z[indices]
        campo_val = magnitudine[indices]

        # Normalizza i valori del campo per la colormap
        if np.max(campo_val) > 0:
            colors = campo_val / np.max(campo_val)
        else:
            colors = np.zeros_like(campo_val)

        # Crea la figura 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot 3D con colori basati sull'intensità del campo
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', alpha=0.6, s=5)

        # Aggiungi la colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Intensità del campo normalizzata')

        # Posizioni dei magneti e sensori
        for i, magnete in enumerate(self.magneti):
            pos = magnete['posizione']
            ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, marker='o',
                      label=f'Magnete {i+1}' if i == 0 else "")

        for i, sensore in enumerate(self.sensori):
            pos = sensore['posizione']
            ax.scatter(pos[0], pos[1], pos[2], color='blue', s=80, marker='s',
                      label=f'Sensore {i+1}' if i == 0 else "")

        # Aggiungi legenda se ci sono elementi
        if self.magneti or self.sensori:
            ax.legend()

        # Etichette degli assi
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_title(f'Campo magnetico 3D al tempo t={tempo} s (soglia = {soglia})')

        # Limiti degli assi
        ax.set_xlim(0, self.dimensioni[0])
        ax.set_ylim(0, self.dimensioni[1])
        ax.set_zlim(0, self.dimensioni[2])

        plt.tight_layout()
        plt.show()

# Funzione di esempio per dimostrare l'uso della classe
def esempio_demo():
    """
    Funzione dimostrativa per simulare un caso d'uso tipico
    """
    print("Inizializzazione del simulatore...")
    simulatore = SimulatoreCampoMagnetico3D(dimensioni_spazio=(100, 100, 100), risoluzione=2.0)

    # Aggiungi alcuni magneti
    simulatore.aggiungi_magnete(posizione=(30, 50, 50), momento_magnetico=(0, 0, 5), frequenza=100)
    simulatore.aggiungi_magnete(posizione=(70, 50, 50), momento_magnetico=(0, 0, -5), frequenza=100)

    # Definisci un'area target (ad esempio, area di un campione da analizzare)
    area_target = ((40, 60), (40, 60), (40, 60))

    # Calcola il campo magnetico
    print("\nCalcolo del campo magnetico iniziale...")
    simulatore.calcola_campo_totale(tempo=0.0)

    # Visualizza il campo magnetico iniziale
    print("\nVisualizzazione del campo magnetico iniziale...")
    simulatore.visualizza_campo(piano='xy', posizione=50)

    # Ottimizza le frequenze per l'area target
    print("\nOttimizzazione delle frequenze per l'area target...")
    simulatore.ottimizza_frequenze(area_target, range_frequenze=(50, 500))

    # Ottimizza le ampiezze per l'area target  # Corretto: chiama ottimizza_ampiezze
    print("\nOttimizzazione delle ampiezze per l'area target...")
    simulatore.ottimizza_ampiezze(area_target)


    # Ricalcola il campo con i parametri ottimizzati
    print("\nRicalcolo del campo magnetico con parametri ottimizzati...")
    simulatore.calcola_campo_totale(tempo=0.0)

    # Visualizza il campo ottimizzato
    print("\nVisualizzazione del campo magnetico ottimizzato...")
    simulatore.visualizza_campo(piano='xy', posizione=50)

    # Trova le posizioni ottimali per i sensori
    print("\nDeterminazione delle posizioni ottimali per i sensori...")
    simulatore.ottimizza_posizioni_sensori(num_sensori=3)

    # Visualizza il campo con i sensori
    print("\nVisualizzazione del campo magnetico con i sensori ottimizzati...")
    simulatore.visualizza_campo_3d(soglia=0.3)

      # Ottimizza le posizioni dei magneti
    print("\nOttimizzazione delle posizioni dei magneti per l'area target...")
    posizioni_ottimali = simulatore.ottimizza_posizioni_magneti(
        num_magneti=2, area_target=area_target
        )

    # Imposta le nuove posizioni dei magneti
    for i, pos in enumerate(posizioni_ottimali):
        if i < len(simulatore.magneti):  # Assicurati di non superare il numero di magneti esistenti
            simulatore.magneti[i]['posizione'] = pos
            print(f"Magnete {i+1} riposizionato a {pos}")

    # Ricalcola e visualizza il campo con le posizioni ottimizzate
    print("\nRicalcolo del campo magnetico con posizioni dei magneti ottimizzate...")
    simulatore.calcola_campo_totale(tempo=0.0)
    print("\nVisualizzazione del campo magnetico con posizioni ottimizzate...")
    simulatore.visualizza_campo(piano='xy', posizione=50)


    return simulatore
# Funzione principale
if __name__ == "__main__":
    print("Simulatore di campo magnetico 3D con accelerazione GPU")
    print("=" * 60)

    # Esegui la demo
    simulatore = esempio_demo()

    print("\nSimulazione completata!")