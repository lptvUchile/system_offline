def Escribir_ctm(Ruido_diff, Evento_diff,Indice,fs):
    

    
    if len(Evento_diff[0]) == len(Evento_diff[1]) and len(Ruido_diff[0]) == len(Ruido_diff[1]):
        if len(Evento_diff[0])==0:
            fs.write(str(0) + ' ' + str(0) + ' EVENTO\n')
    
    
        for i in range(len(Evento_diff[0])):
            Inicio = 2*Evento_diff[0][i]
            Duracion = (Evento_diff[1][i]-Evento_diff[0][i]+1)*2
            fs.write(str(Inicio) + ' ' + str(Duracion) + ' EVENTO\n')

