import pandas as pd


local="/home/alexandre/Documentos/Artigo/Random Florest/Dados/Tabulação.csv"
dados=pd.read_csv(local)
dados=dados.drop(columns=['99'])
estatistica=dados.describe()
############# Dados
dados['Motricidade_fina']=pd.get_dummies(dados['Motricidade_fina'])['Consegue']
dados['Motricidade_grosa']=pd.get_dummies(dados['Motricidade_grosa'])['Consegue']
######Equilibrio 
dados['Equilíbrio']=pd.get_dummies(dados['Equilíbrio'])['Consegue']
#####Esq_corp_mãos
dados['Esq_corp_mãos'][dados['Esq_corp_mãos']=="Consegue"]=1
dados['Esq_corp_mãos'][dados['Esq_corp_mãos']=="Não consegue"]=0
#####Esq_corp_braço
dados["Esq_corp_braços"][dados["Esq_corp_braços"]=="Consegue"]=1
dados["Esq_corp_braços"][dados["Esq_corp_braços"]=="Não consegue"]=0
######Organização_Espacial
dados["Organização_espacial"][dados["Organização_espacial"]=="Consegue"]=1
dados["Organização_espacial"][dados["Organização_espacial"]=="Não consegue"]=0
######Lateralidade_mãos_Direita
#dados['Lateralidade_mãos_Direita']=pd.get_dummies(dados['Lateralidade_mãos'])['Direita']
######Lateralidade_mãos_Esquerda
#dados['Lateralidade_mãos_Esquerda']=pd.get_dummies(dados['Lateralidade_mãos'])['Esquerda']
######]Lateralidade_mãos_ambos
#dados['Lateralidade_mãos_ambos']=pd.get_dummies(dados['Lateralidade_mãos'])['D/E']
#######Lateralidade_olhos_Direita
#dados['Lateralidade_olhos_Direita']=pd.get_dummies(dados['Lateralidade_olhos'])['Direita']
#######Lateralidade_olhos_Esquerda
#dados['Lateralidade_olhos_Esquerda']=pd.get_dummies(dados['Lateralidade_olhos'])['Esquerda']
#######Lateralidade_olhos_Ambos
#dados['Lateralidade_olhos_Ambos']=pd.get_dummies(dados['Lateralidade_olhos'])['D/E']
######Lateralidade_pés_Direito
#dados['Lateralidade_pés_Direito']=pd.get_dummies(dados['Lateralidade_pés'])['Direita']
#####Lateralidade_pés_Esquerda
#dados['Lateralidade_pés_Esquerdo']=pd.get_dummies(dados['Lateralidade_pés'])['Esquerda']
#####Idade
#dados['Idade_1.0']=pd.get_dummies(dados['Idade'])['1.0']
dados['Idade_2.0']=pd.get_dummies(dados['Idade'])[2.0]
dados['Idade_3.0']=pd.get_dummies(dados['Idade'])[3.0]
#dados['Idade_4.0']=pd.get_dummies(dados['Idade'])['4.0']
#dados['Idade_5.0']=pd.get_dummies(dados['Idade'])['5.0']
#####Graffar
#dados['Graffar 1']=pd.get_dummies(dados['Graffar'])['Classe 1']
#dados['Graffar 2']=pd.get_dummies(dados['Graffar'])['Classe 2']
#dados['Graffar 3']=pd.get_dummies(dados['Graffar'])['Classe 3']
#dados['Graffar 4']=pd.get_dummies(dados['Graffar'])['Classe 4']
#dados['Graffar 5']=pd.get_dummies(dados['Graffar'])['Classe 5']
#####Profissão
dados['Profissão 1']=pd.get_dummies(dados['Profissão'])['Grau 1']
#dados['Profissão 2']=pd.get_dummies(dados['Profissão'])['Grau 2']
dados['Profissão 3']=pd.get_dummies(dados['Profissão'])['Grau 3']
dados['Profissão 4']=pd.get_dummies(dados['Profissão'])['Grau 4']
dados['Profissão 5']=pd.get_dummies(dados['Profissão'])['Grau 5']
#######Instrução
dados['Instrução 1']=pd.get_dummies(dados['Instrução'])['Grau 1']
dados['Instrução 2']=pd.get_dummies(dados['Instrução'])['Grau 2']
dados['Instrução 3']=pd.get_dummies(dados['Instrução'])['Grau 3']
dados['Instrução 4']=pd.get_dummies(dados['Instrução'])['Grau 4']
dados['Instrução 5']=pd.get_dummies(dados['Instrução'])['Grau 5']
########Fonte_Renda
#dados['Profissão 1']=pd.get_dummies(dados['Profissão'])['Grau 1']
#dados['Renda 2']=pd.get_dummies(dados['Fonte_renda'])['Grau 2']
dados['Renda 3']=pd.get_dummies(dados['Fonte_renda'])['Grau 3']
dados['Renda 4']=pd.get_dummies(dados['Fonte_renda'])['Grau 4']
dados['Renda 5']=pd.get_dummies(dados['Fonte_renda'])['Grau 5']
########Tipo_Habitação
#dados['Tipo_habitação 1']=pd.get_dummies(dados['Tipo_habitação'])['Grau 1']
dados['Tipo_habitação 2']=pd.get_dummies(dados['Tipo_habitação'])['Grau 2']
dados['Tipo_habitação 3']=pd.get_dummies(dados['Tipo_habitação'])['Grau 3']
dados['Tipo_habitação 4']=pd.get_dummies(dados['Tipo_habitação'])['Grau 4']
#dados['Tipo_habitação 5']=pd.get_dummies(dados['Tipo_habitação'])['Grau 5']
########Local_residência
#dados['Local_residência 1']=pd.get_dummies(dados['Local_residência'])['Grau 1']
dados['Local_residência 2']=pd.get_dummies(dados['Local_residência'])['Grau 2']
dados['Local_residência 3']=pd.get_dummies(dados['Local_residência'])['Grau 3']
dados['Local_residência 4']=pd.get_dummies(dados['Local_residência'])['Grau 4']
#dados['Local_residência 5']=pd.get_dummies(dados['Local_residência'])['Grau 5']

########Resultado
#dados['Resultado 1']=pd.get_dummies(dados['Resultado'])['Classe I']
dados['Resultado 2']=pd.get_dummies(dados['Resultado'])['Classe II']
dados['Resultado 3']=pd.get_dummies(dados['Resultado'])['Classe III']
dados['Resultado 4']=pd.get_dummies(dados['Resultado'])['Classe IV']
#dados['Resultado 5']=pd.get_dummies(dados['Resultado'])['Classe V']

dados.corr().to_csv("/home/alexandre/Documentos/Artigo/Random Florest/Dados/Correlograma.csv")
print(len(dados.columns))
dados=dados.drop(columns=['Lateralidade_mãos','Lateralidade_olhos','Lateralidade_pés','Idade','Graffar','Profissão','Instrução' ,'Fonte_renda','Tipo_habitação',	'Local_residência','Resultado'])
dados.to_csv("/home/alexandre/Documentos/Artigo/Random Florest/Dados/dados_limpos.csv")
