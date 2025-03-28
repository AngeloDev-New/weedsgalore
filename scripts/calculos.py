import numpy as np

class Globo:
    def __init__(self, *localizacoes):
        mDis, mPar = self.maior_distancia(*localizacoes)
        self.diametroGlobo = mDis
        self.bordas = mPar
        self.excludes = []

    def insideGlob(self, dist):
        distance, _ = self.maior_distancia(dist, *self.bordas)
        return distance <= self.diametroGlobo
    
    def getClass(self, point):
        for i, glob in enumerate(self.excludes):
            if glob.insideGlob(point):
                return i + 2
        return 1 if self.insideGlob(point) else 0

    @staticmethod
    def distancia_euclidiana(*dimensoes):
        return np.sqrt(sum(np.square(dim[0] - dim[1]) for dim in dimensoes))
    
    def maior_distancia(self, *localizacoes):
        distancia_maior = 0
        par_maior = None
    
        for i, localA in enumerate(localizacoes):
            for j, localB in enumerate(localizacoes):
                if i < j:  # Evita calcular duas vezes a mesma distância
                    distancia = self.distancia_euclidiana(*zip(localA, localB))
                    if distancia > distancia_maior:
                        distancia_maior = distancia
                        par_maior = (localA, localB)
    
        return distancia_maior, par_maior  # Retorna a maior distância e o par correspondente
    
    def exclude(self, glob):
        self.excludes.append(glob)

if __name__ == '__main__':
    # Pontos iniciais para definir o Globo
    a = (0, 0, 0)
    b = (10, 10, 10)
    c = (5, 5, 5)
    globo = Globo(a, b, c)
    
    # Criar um segundo globo para teste de exclusão
    d = (2, 2, 2)
    e = (4, 4, 4)
    sub_globo = Globo(d, e)
    globo.exclude(sub_globo)
    
    # Teste de pontos dentro e fora dos Globos
    dentro = (5, 5, 4)  # Ponto dentro do Globo
    fora = (15, 15, 15)  # Ponto fora do Globo
    dentro_sub = (3, 3, 3)  # Ponto dentro do sub_globo
    
    print("Maior distância e bordas do Globo:", globo.diametroGlobo, globo.bordas)
    print("O ponto", dentro, "está dentro do Globo?", globo.insideGlob(dentro))
    print("O ponto", fora, "está dentro do Globo?", globo.insideGlob(fora))
    print("O ponto", dentro_sub, "pertence a qual classe?", globo.getClass(dentro_sub))
