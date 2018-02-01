import math

GLOBAL_SOL_CONST = 1355 # W/m2

def getSolarBreakDown(totalSol, solAlt):
	"""
		totalSol: float, in W/m2
		solAlt: float, in degree
	"""
	# Watanabe method
	# Dir sol
	if (totalSol>0):
		kt = getKt(totalSol, solAlt);
		ktc = getKtc(solAlt);
		kds = getKds(kt, solAlt, ktc);
		dh = GLOBAL_SOL_CONST*math.sin(math.radians(solAlt))*kds*(1-kt)/(1-kds);
		sh = GLOBAL_SOL_CONST*math.sin(math.radians(solAlt))*(kt-kds)/(1-kds);
		if (dh>0):
			dirSol = dh;
		else:
			dirSol = 0.0;
		if (sh>0):
			difSol = sh;
		else:
			difSol = 0.0;
			
	else:
		dirSol = 0.0;
		difSol = 0.0;
	return (dirSol, difSol);
	

def getKt(totalSol, solAlt):
	return totalSol/(GLOBAL_SOL_CONST*math.sin(math.radians(solAlt)));

def getKtc(solAlt):
	return 0.4268 + 0.1934*math.sin(math.radians(solAlt));

def getKds(kt, solAlt, ktc):
	if (kt>=ktc):
		return (kt-(1.107+0.03569*math.sin(math.radians(solAlt)) +
					1.681*math.pow(math.sin(math.radians(solAlt)), 2))*
					math.pow((1-kt), 2));
	else:
		return ((3.996-3.862*math.sin(math.radians(solAlt))+
					1.540*math.pow(math.sin(math.radians(solAlt)), 2))*
					math.pow(kt, 3));
		

