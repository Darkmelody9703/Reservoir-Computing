'''
2023年8月26日17:22:19
'''
import cma
import torch
from main_sparch_better import *

model = torch.load(r'log\08-26-02-53\ckpt\best_model_1_92.57950530035336.pth').to('cuda:0')
train_loader, test_loader = load_shd_or_ssc()

best_alpha = [0.8243803092264532, 0.9604688603348298, 0.9607846703736647, 0.9595186519350822, 0.8490250989289092, 0.8982046365206268, 0.8201879182077841, 0.8886192072579968, 
              0.9607894354419935, 0.8187661417337029, 0.8260448444187579, 0.8399187800296382, 0.8214084888649013, 0.9598658927126352, 0.824349687270403, 0.9194435184685358, 
              0.9605387153559588, 0.9607010715119774, 0.8266095465352128, 0.9607363135861428, 0.8579427964912295, 0.9002061419258588, 0.9571926900478269, 0.9606909771316208, 
              0.9607762075393097, 0.8199289003384398, 0.8189059687186506, 0.9505890861364736, 0.8199446143072293, 0.8445779569120768, 0.9437389928939655, 0.9604030462280556, 
              0.9494523653064589, 0.9607602505150685, 0.9607581572980007, 0.8521701837729728, 0.8230441386404667, 0.8230912159234757, 0.960529276155468, 0.9607866810730222, 
              0.9602192847257871, 0.8963851844853313, 0.8738005598439115, 0.9364583201456985, 0.819067752127089, 0.8215041011272918, 0.819132352790682, 0.8347785639352913, 
              0.8207016327137593, 0.8194987992354303, 0.9587952227120866, 0.9598550097240957, 0.903877258877084, 0.9603680844356108, 0.9607388099926344, 0.821362228976546, 
              0.9607893967339695, 0.954899653856994, 0.9518757826554579, 0.9525813422233127, 0.9595204153437529, 0.8242856229413854, 0.8280071856024558, 0.9606934047639425, 
              0.8685589062658915, 0.960412648222209, 0.9514077919457887, 0.9562646172275597, 0.9574630427082262, 0.9328639043688526, 0.917351716135774, 0.960430357336817, 
              0.9606556389879295, 0.9352913242082901, 0.9607409322386806, 0.9589526034701341, 0.9607428813482498, 0.9602265899664808, 0.960341049475823, 0.9606446746191049, 
              0.8193020943231905, 0.8573900350225202, 0.8187491041749018, 0.9467894032435651, 0.8188387023306377, 0.8312712158197973, 0.9387791697529836, 0.9590466028581536, 
              0.8834143822832881, 0.9510179565586351, 0.9231224303804877, 0.8907084574145179, 0.9606951075022662, 0.9572023360042495, 0.9603643129489369, 0.8191476691745071, 
              0.9607491415612329, 0.8187333916156166, 0.8197813769198975, 0.8188362316849725, 0.8285907009424582, 0.9606890454945898, 0.8760761226097046, 0.8193046575237505, 
              0.9592829885591941, 0.9606736543764619, 0.9575017411549497, 0.8190673273823181, 0.9580339205211114, 0.9335687218163271, 0.8966046808871915, 0.960009431960733, 
              0.9401746504293228, 0.9599830826906847, 0.8509513892207039, 0.8187638426666628, 0.8191720811240033, 0.8527395391292375, 0.8279726713621344, 0.8321640127750576, 
              0.9607872288087851, 0.9604206068584926, 0.8211383019152093, 0.8518846391361583, 0.8188291645499794, 0.9501197753878633, 0.9518566633074078, 0.956313237815687]

best_beta = [0.97256395, 0.99074706, 0.9672161, 0.99029245, 0.96814693, 0.99165949, 0.96955689, 0.96927495, 0.99110786, 0.99165544,
       0.99085009, 0.98046028, 0.99166628, 0.99164499, 0.9916924, 0.99060472, 0.96775346, 0.98845951, 0.9672264, 0.99145321,
       0.98186853, 0.96738463, 0.98292404, 0.99075785, 0.98798603, 0.97446569, 0.99167624, 0.98779759, 0.99156251, 0.98920214,
       0.99165829, 0.96939714, 0.96869089, 0.9726575 , 0.96740402, 0.99031122, 0.99138038, 0.96943011, 0.96833877, 0.9674483 ,
       0.98985584, 0.97008662, 0.98570504, 0.99150564, 0.99169906, 0.98066576, 0.96752089, 0.96724828, 0.96951252, 0.97217376,
       0.99050927, 0.9678066, 0.9772856 , 0.98489694, 0.99165317, 0.99168238, 0.98854559, 0.99113124, 0.99121836, 0.96876005,
       0.99096571, 0.96858371, 0.99165413, 0.96732699, 0.9916783, 0.99168708, 0.98863382, 0.99168978, 0.99170129, 0.98845998,
       0.9897373, 0.99169406, 0.9909825 , 0.97718075, 0.96753425, 0.98701238, 0.98104809, 0.9903783 , 0.97482432, 0.96722926,
       0.99170129, 0.99163632, 0.9907788 , 0.98952895, 0.99149605, 0.96835563, 0.99161299, 0.99140264, 0.96760448, 0.98630127,
       0.96733564, 0.99151831, 0.99086999, 0.98700407, 0.99149767, 0.99040365, 0.97888822, 0.98709541, 0.99099017, 0.9895705 ,
       0.99158849, 0.97452827, 0.99081035, 0.99091797, 0.96723802, 0.98764014, 0.96737677, 0.99164076, 0.98214805, 0.97035564,
       0.96745987, 0.97133753, 0.99071728, 0.99057185, 0.97148519, 0.99166722, 0.99040249, 0.99081736, 0.96790447, 0.98835052,
       0.99044821, 0.96959204, 0.98342748, 0.96923212, 0.99163885, 0.99063316, 0.9905511 , 0.99100212]

def objective_func(x):
    a = x[0: 128]
    b = x[128: 128*2]
    model.eval()
    with torch.no_grad():
        for i in range(128):
            model.alpha.data[i*8:(i+1)*8] = torch.tensor(best_alpha[i]).cuda() # *(np.exp(-1 / 25) - np.exp(-1 / 5)) + np.exp(-1 / 5)
            model.beta.data[i*8:(i+1)*8] = torch.tensor(best_beta[i]).cuda()
            model.a.data[i*8:(i+1)*8] = torch.tensor(a[i]).cuda()
            model.b.data[i*8:(i+1)*8] = torch.tensor(b[i]).cuda()
        losses, correct, total = [], 0, 0
        for images, labels in test_loader:
            images = torch.sign(images.clamp(min=0)) # all pixels should be 0 or 1
            outputs, firing_rates, all_spikes = model(images.to(config.device), 0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total

    return 93.7279151943463-accuracy

# 定义初始参数和每个参数维度上的步长
# x0 = np.random.uniform(np.exp(-1/5), np.exp(-1/25), size=(128))  # 初始参数值
x0 = np.concatenate((model.a.data.cpu().view(8,128).mean(0).numpy(),
                     model.b.data.cpu().view(8,128).mean(0).numpy()))
# x0 = np.concatenate((np.random.uniform(np.exp(-1/5), np.exp(-1/25), 128), 
                # np.random.uniform(np.exp(-1/30), np.exp(-1/120), 128),
                # np.random.uniform(-1, 1, 128),
                # np.random.uniform(0, 2, 128),))
sigma0 = 0.01  # 参数步长

# 定义参数的上下界
lower_bounds = [-1]*128 + [0]*128
upper_bounds = [1]*128 + [2]*128
# lower_bounds = [np.exp(-1/5)]*128 + [np.exp(-1 / 30)]*128 + [-1]*128 + [0]*128  # 参数的下界
# upper_bounds = [np.exp(-1/25)]*128 + [np.exp(-1 / 120)]*128 + [1]*128 + [2]*128  # 参数的上界
bounds = [lower_bounds, upper_bounds]

# 创建CMAES对象并运行优化
es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': bounds,
                                           'popsize': 50,
                                           'maxiter': 1000,
                                          #  'verbose': True,
                                           'verb_disp': 1,
                                           'tolfun': 1e-11, 
                                           'tolstagnation': 50})
es.optimize(objective_func)

# 输出最佳解和对应的目标函数值
best_solution = es.result.xbest
best_fitness = es.result.fbest
print("Best solution found: ", best_solution)
print("Best fitness value: ", best_fitness)