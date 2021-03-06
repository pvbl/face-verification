import sklearn.preprocessing as pr
from sklearn.compose import ColumnTransformer
from sklearn import set_config
from sklearn.pipeline import Pipeline

# to transformer
def toTransformer(func):
    return pr.FunctionTransformer(func)



class TranformationCategorical(object):
    def __init__(self,columns,pipeline=[]):
        self.__tmp__pipeline = pipeline
        self.pipeline = pipeline
        self.columns= columns
    def __ap(self,step):
        self.__tmp__pipeline.append(step)
    def addlabelencoding(self,**args):
        self.__ap(("labenc",pr.LabelEncoder(**args)))
    def addordinalencoding(self,classes,**args):
         self.__ap(("ordenc",pr.LabelBinarizer(classes=classes,**args)))
    def addbinningEncoding(self,**args):
        return None
    def addOHE(self,**args):
        self.__ap(("ohe",pr.OneHotEncoder(**args)))
    def buildColumnTranformer(self,name):
        steps=(name,Pipeline(self.__tmp__pipeline),self.columns)
        ctrans = ColumnTransformer(steps)
        self.__tmp__pipeline = []
        return self.pipeline.append(ctrans)
    def returnSteps(self):
        return self.pipeline
    def plot_pipeline(self,type="tmp"):
        set_config(display="diagram")
        print(self.pipeline)


class TranformationContinuous(object):
    def __init__(self,pipeline=[]):
        self.__tmp__pipeline = pipeline
        self.pipeline = pipeline
    def sweked_boxCox(self,column):
        return pr.PowerTransformer(method='box-cox', standardize=True, copy=True)
    def sweked_YeoJohnson(self,column):
        return pr.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
    def scaling(self,values,max=None,min=None):
        return None
    def normalization_zscore(self,column):
        return None
    @toTransformer
    def logp1(self,values):
        return np.log1p(values)
    @toTransformer
    def log(self,values):
        return np.log(values)
    def meancentering(self,values):
        return values-np.mean(values)
    def robustscaling(self,values):
        return None
    def convert_to_polinomial(self,**args):
        return PolynomialFeatures(**args)
    def binningEncodingQuantiles(self,values,quantiles=5):
        return (pr.QuantileTransformer(),column)

class PipelineBuild(object):
    def __init__(self):
        pass



@hp.toTransformer
def winsorization(series,quantile=[0,0.99],overzero=False):
    assert len(quantile)==2, ValueError("allowed only a quantile list of 2 values, min and max")
    s= series.copy()
    if overzero:
        quant_low,quant_high = s[s>0].quantile(quantile).sort_values().values
    else:
        quant_low,quant_high = s.quantile(quantile).sort_values().values
    s.loc[s <= quant_low] = quant_low
    s.loc[s >= quant_high] = quant_high
    return s



class TransformationsLatLon(object):
	def manhattan_distance(self,lat,lon):
		return None
	def haversine_distance(self,lat,lon):
		return None
	def bearing(self,lat,lon):
		return None
