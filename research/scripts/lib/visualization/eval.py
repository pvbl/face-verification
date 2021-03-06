import matplotlib.pyplot as plt

def show_plot(iteration,loss,**args):
	plt.plot(iteration,loss,**args)
	plt.show()
