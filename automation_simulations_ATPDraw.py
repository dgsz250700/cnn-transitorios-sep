import os
import csv
import random
import scipy.io as sio

# Configuraciones iniciales
folder = "C:\\emtp\\atpdraw\\results\\Banco_csv"
cuantas = int(input("Ingrese número de muestras: "))
filename = "test_banco.csv"
ruta = os.path.join(folder, filename)
voltajes = []

# Proceso repetitivo para obtener diferentes vectores de voltaje
for i in range(cuantas):
    archivo_original = "C:\\emtp\\atpdraw\\results\\bancotesis.atp"
    vieja = ".03333"
    n = random.randint(2500, 9999)
    nueva = ".0" + str(n)
    archivomod = "C:\\emtp\\atpdraw\\results\\bancotesis1p.atp"

    # Modificación del archivo
    with open(archivo_original, 'r', encoding='latin-1') as f_orig, open(archivomod, 'w', encoding='latin-1') as f_mod:
        for linea in f_orig:
            linea_modificada = linea.replace(vieja, nueva)
            f_mod.write(linea_modificada)

    # Comandos del sistema para procesar el archivo
    os.system(r'"C:\ATP\atpmingw\tpbig.exe" disk C:\emtp\atpdraw\results\bancot