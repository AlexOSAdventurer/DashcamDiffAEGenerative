import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import diffusion_model as df
import video_model
import subprocess
import sys

class VideoPredictionModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_model = df.DiffusionModel_load_pretrained().eval()
        self.transformer_model = video_model.TransformerModel(
            config["model"]["ninp"],
            config["model"]["nhead"],
            config["model"]["nhid"],
            config["model"]["nlayers"],
            config["model"]["dropout"]
        )
        self.val_batch = None
        self.latent_mean_orig = torch.FloatTensor([-3.66955971e-03, -5.07708741e-03, -1.54310086e-02,  6.94912811e-03,
           -3.32588424e-04,  1.52487159e-02,  2.28158367e-03, -1.75829519e-03,
            1.51329820e-03, -2.97139135e-04, -3.82107584e-03,  3.13244181e-03,
           -7.06045120e-03,  5.76423638e-03, -1.89320766e-02, -4.15670969e+00,
            8.81587208e-03, -4.48908867e+00, -1.16093849e-02,  1.96186312e-04,
           -1.91318368e-02, -3.76094229e+00,  8.63385325e-03, -1.54471542e-01,
            1.79620451e-02, -2.22316197e-02, -2.63664454e-03, -1.46965427e-02,
           -3.68455050e-03, -1.48457019e-02, -2.47588475e-02, -1.97703008e-02,
           -5.80541538e-03,  4.68774575e-03,  1.71752409e-04, -6.22809278e-03,
            4.26428544e-03, -1.25535261e-02,  1.28419610e-02,  9.28814574e-03,
            5.61059613e-03,  3.21796578e-02,  1.99450308e-02,  1.16766515e-02,
            3.14701921e-03, -1.77554976e-02,  8.58248942e-03, -1.10053124e-02,
           -2.04063998e-02, -1.54653293e-02, -1.69199425e-02,  1.51163382e-04,
           -6.82500486e-03,  2.97907338e-03,  2.43290105e-02, -3.62762691e-02,
            1.09613809e-02,  1.76917720e-02, -2.41770818e-03, -1.75906762e-02,
            1.28572019e-02, -4.37373163e-03,  4.44055626e-03, -1.55342932e-02,
            8.90593862e-03, -1.48914588e-02, -2.40705167e-02, -5.89633661e-03,
            2.74898328e-02, -1.09428055e-03,  1.28958254e-02,  6.47193175e-03,
           -8.15131140e-03, -4.39107709e-03,  1.18542454e-02, -1.13231827e-02,
           -3.06528368e+00,  7.87554209e-03, -4.55229794e+00,  1.42763948e-02,
            2.63137363e-02,  6.86344107e-03, -4.56640632e+00, -3.16935172e-02,
            1.58481136e-02, -5.76334949e-01,  2.10095593e-03, -9.87649928e-03,
           -1.05849708e-02, -1.25748163e-02, -1.81556431e-02, -2.74127079e+00,
           -3.96641519e-03,  1.24177053e-02, -1.22782795e-02,  1.70090256e-02,
           -4.40398716e-03, -4.87740924e+00,  2.54675633e-03,  1.00053308e-03,
           -5.33327113e+00,  9.17217097e-03,  1.65479791e-03,  1.57345655e-02,
           -4.18555548e+00, -1.06800473e-02,  2.18806567e-03,  9.56339164e-03,
            1.79192519e-02, -8.38175092e-03, -4.81371634e-03,  1.48473314e-02,
           -1.34588821e-02,  1.56436889e-02, -8.00145226e-03, -5.58365712e-03,
           -2.59739980e-02,  3.43191189e-04, -3.08526737e-02, -4.28729037e+00,
            7.61584334e-03,  1.04240633e-02, -4.58701111e+00,  2.23299400e-02,
           -4.95634680e+00, -2.15275397e-02,  1.59634699e-02,  1.08491069e-02,
           -1.31040345e-02,  1.67030839e-02,  1.38516045e-03, -1.24533773e-02,
            1.00208864e-02, -2.49481537e-02,  8.13912662e-04,  1.81178045e-02,
           -1.37079460e-03,  2.58325083e-02, -4.67999161e+00,  1.23020968e-03,
           -6.69272589e-02,  9.81505848e-03, -1.73594829e-02, -5.57625842e-03,
            1.39969203e-02, -5.62617003e-03, -1.11562639e-02, -3.20518395e-02,
           -2.24838658e-02, -4.65128840e+00,  1.78840484e-02,  1.91846897e-02,
           -7.51844229e-03,  3.36064665e-04, -2.24709673e-02, -1.88465309e-02,
           -7.91723168e-03, -2.77445303e-02, -8.47152090e-01, -2.92683579e-02,
            4.31421500e-02, -2.37987417e-02,  8.66568919e-03, -4.88845909e-03,
            1.74935921e-02,  1.30949986e-04, -3.98824378e-03,  9.04225912e-03,
           -2.37862265e-02,  3.78300324e-03,  2.55646307e-03,  5.95054603e-03,
           -4.06230241e-02,  5.56118540e-03, -9.46229165e-03,  9.73994972e-03,
            2.11872780e-02,  1.64001359e-02, -2.18561872e-02,  1.97585928e-04,
            2.18464832e-02, -4.44747587e-03, -1.29581575e-02,  1.74064404e-03,
            6.34032809e-03, -9.54406367e-03, -3.71797356e-03,  1.52620748e-02,
           -1.22303900e-02, -9.68168934e-03, -4.86576089e+00,  1.31979105e-02,
            1.54624214e-03, -3.92147410e+00,  2.36595939e-02, -1.95692916e-02,
            5.31347726e-03,  1.87244818e-02, -3.40350657e-02,  3.10250049e-04,
           -1.08744781e-03, -4.15121081e+00,  6.14833136e-03,  2.44539273e-03,
            8.28274964e-03,  6.49753520e-03,  5.47294152e-03, -1.73652396e-02,
           -3.84327898e-02, -9.73238825e-03, -4.60717286e+00, -4.74558857e-02,
           -1.19424157e-02,  7.10824099e-03,  6.40473076e-03, -2.11353556e-02,
            1.91131520e-02,  7.66881306e-04, -1.46447398e-02,  1.29369922e-02,
           -2.00134039e-02,  3.30435628e-03,  1.84055320e-02,  3.89340183e-03,
           -4.64259624e+00,  2.29198887e-02,  2.87845461e-03,  1.51460633e-02,
           -2.44414203e-02, -3.74934540e-03, -5.01727891e-03,  9.47079966e-04,
            1.69594954e-02,  5.69002161e-03,  2.91309308e-03,  1.56226285e-02,
           -4.16752316e-03, -1.58386893e-01, -9.82350231e-02,  1.12429702e-02,
           -6.97373979e-04,  6.88126014e-03, -2.13199320e-03,  1.92818263e-03,
            2.06179957e-03, -8.57209450e-03, -3.31181875e-03, -6.67403019e-01,
           -9.23136581e-03,  1.20526031e-02, -2.81484802e-02,  1.18445443e-02,
           -5.05855177e-03,  1.87554296e-02,  1.92792641e-02, -4.23780209e+00,
           -4.44714037e-02, -1.20635717e-02,  8.92357937e-03, -8.91663529e-03,
           -2.24648439e-02,  9.40018325e-03,  1.17767753e-02, -5.14537739e+00,
            1.76696406e-02, -5.33229606e+00, -2.27246521e-02, -3.37277273e-03,
            3.47067209e-03,  9.21149469e-03, -4.76833843e-03,  1.72628330e-02,
            4.98565540e-03, -2.20362959e-04, -1.49211699e-02,  5.55592116e-03,
           -4.33522745e-02,  6.35665128e-03,  1.92356811e-02,  1.31292600e-02,
           -9.24816176e-03, -2.26860200e-02, -1.92856622e-02, -9.90399024e-03,
            1.99978500e-02, -4.44134302e+00, -6.81322693e-04,  3.50848428e-02,
           -2.18081345e-02,  4.10139269e-03,  1.13680038e-02, -3.25651414e-02,
            1.72499327e-02,  4.99532698e-03,  6.18449749e-03,  1.73368637e-02,
           -6.46477964e-03, -3.50658259e-04,  1.97187155e-03, -9.83228494e-03,
            2.98780018e-02, -3.27680806e-02, -2.80512392e-03,  1.72505390e-04,
           -1.33452739e-02, -5.01782534e+00,  5.33117025e-03,  5.02603056e-03,
           -1.58621655e-02, -1.69989224e-02, -2.40269246e+00, -1.19865201e-04,
           -3.51297029e-03, -7.58746269e-04,  2.10082146e-02, -6.92212210e-03,
           -1.64535656e-02, -2.18304147e-02, -6.34805893e-03, -2.70183892e-02,
            2.71787121e-02,  3.48128268e-02, -8.19006026e-03, -4.11805087e+00,
           -2.10932550e-02, -4.38903556e-03, -6.22545750e-03, -4.44343055e-03,
           -1.25144399e-02,  1.14262068e-02, -1.67455620e-02, -1.24627824e-02,
           -4.85530217e-05, -4.18011955e-01, -6.97945939e-01, -4.60897702e-03,
            2.26037760e-02,  5.65780274e-03,  2.22230743e-02,  1.89883233e-02,
            1.33077166e-02,  1.36736396e-02,  1.41474066e-02,  9.23307283e-03,
           -3.45473313e-02, -1.35025767e-02, -1.55889632e-03, -6.61951293e-03,
           -1.10297223e-02, -4.61550482e+00, -8.61183007e-03, -7.27371380e-02,
            1.26852367e-02,  3.70062593e-03,  5.67945984e-03, -4.69041302e+00,
            8.84386099e-03, -8.31884482e-03,  3.12884339e-03, -1.62184175e-02,
           -1.64151942e-02, -4.70376328e+00, -4.33237688e+00, -4.29564998e-02,
            6.84401540e-03, -6.94001704e-03, -5.77625890e-02, -1.85850076e-03,
           -7.64780574e-03, -4.93472834e+00,  7.07782837e-03,  2.49990084e-03,
           -1.20008049e-03,  1.54478415e-02, -1.40687013e-03, -1.23518170e-02,
           -8.12486955e-03,  1.69772926e-02, -4.44357309e-02, -3.17665269e-03,
           -5.08426451e+00, -1.11736619e-01, -1.14460425e-02,  5.47627566e-03,
           -4.61486518e+00, -4.56864201e-05, -1.72978712e-02,  1.34183228e-02,
           -4.46082107e-03,  5.43517352e-03,  4.84113172e-03,  1.27508126e-03,
           -4.52303031e+00,  4.68472374e-03, -1.77738939e-02,  5.42106738e-03,
           -1.69701606e-02,  7.25919774e-03,  1.68306842e-02, -3.03471556e-02,
           -2.43695204e-02, -9.80561489e-03,  1.51622466e-02, -6.64301093e-03,
            2.40644944e-02,  1.88999097e-02, -1.48766723e+00, -3.29492944e-02,
           -1.47447801e-02, -3.59282060e-02,  1.07063201e-02, -3.00439509e-02,
            2.12697490e-02, -1.70373504e-02,  5.75236454e-03, -1.16895646e-02,
           -4.94642676e-02, -4.19383536e-03, -1.24193965e-02, -9.79925205e-03,
           -5.76866868e-03, -1.77843007e-02,  3.28122965e-03,  1.40962657e-02,
            9.50306825e-03,  6.23363802e-03,  5.30910909e-03,  6.37416666e-03,
           -2.20257226e-04, -1.67008846e-02,  7.90096273e-03, -3.25139614e-01,
           -1.00321493e-02, -4.05715458e+00,  2.27459329e-02, -1.71620055e-02,
           -2.01436994e-02,  2.61351032e-03,  1.34802342e-02, -5.02941499e-03,
           -4.41577206e-03, -9.87132063e-01,  6.42535859e-04, -9.91778676e-03,
           -8.09414272e-03,  2.77237061e-03, -1.97226699e-02, -3.35614068e-02,
           -5.29860750e-03,  4.02076774e-03,  1.60355947e-02, -1.74653002e-03,
           -4.17506596e+00,  8.42296374e-03,  1.32373481e-02,  3.10256628e-03,
            2.06492493e-02,  1.64766265e-02, -7.16850073e-03, -1.34360063e+00,
           -9.78700179e-03, -3.51176681e-03, -2.18440396e-02, -1.63057354e-02,
           -5.25495971e-03, -8.68023526e-03, -2.18871699e-02,  9.79272132e-03,
            1.11184129e-02, -4.62791014e+00,  2.19724736e-03, -4.15475723e-03,
            1.23399849e-02,  1.32801641e-04, -2.68735073e-03,  6.91324446e-03,
           -4.88490476e-03, -8.66673675e-03,  2.84617242e-03, -2.72687402e-04,
           -1.93664471e-02,  1.95332370e-02, -4.55898658e-03,  2.13216848e-02,
            6.00309152e-03, -5.85654571e-03, -5.04547150e-03,  1.60416575e-03,
            9.52571579e-03, -1.19619080e-01,  5.22823606e-03, -4.04854011e-02,
           -2.49106547e-02, -7.30236979e-03, -1.98173272e-02, -2.58950794e-03,
           -3.38526910e-02,  1.83297931e-02, -3.53453628e-02, -5.12205941e-02,
           -4.43885693e+00, -2.34433612e-02, -5.99928142e-02,  1.31478619e-02,
           -5.78858459e-03, -1.58883212e-02, -4.12139239e-03,  4.36972951e-03,
           -7.23212395e-03, -1.73020941e-02, -2.62294393e-02,  8.84932713e-03])
        self.latent_std_orig = torch.FloatTensor([0.08395306, 0.08711248, 0.07886596, 0.07855958, 0.08273369,
           0.09149951, 0.08528089, 0.08332507, 0.08402582, 0.08379203,
           0.08057514, 0.08407275, 0.08325371, 0.08666229, 0.07890504,
           0.53816577, 0.08529864, 0.75607972, 0.08183151, 0.08324503,
           0.08263605, 0.65452153, 0.09984779, 0.26620206, 0.08217738,
           0.15560968, 0.08497575, 0.08300066, 0.08192948, 0.08656369,
           0.08059033, 0.08963944, 0.08261694, 0.08364034, 0.08452808,
           0.08476675, 0.08158035, 0.08195475, 0.08271266, 0.0864293 ,
           0.07924113, 0.08633682, 0.08653328, 0.0837578 , 0.08435896,
           0.08294643, 0.08231125, 0.0836366 , 0.08313702, 0.08109325,
           0.08311574, 0.08343671, 0.08482133, 0.08694879, 0.08412524,
           0.07811475, 0.08227561, 0.08516156, 0.08387182, 0.08032794,
           0.08394479, 0.08383866, 0.08720005, 0.07906222, 0.08816742,
           0.08002343, 0.07792603, 0.08898114, 0.08471284, 0.08803619,
           0.08383338, 0.08420776, 0.08224925, 0.08452792, 0.0880391 ,
           0.08300606, 0.64970665, 0.08103635, 0.82409167, 0.08625981,
           0.13326213, 0.08073214, 0.91810742, 0.07941416, 0.08336557,
           0.51525722, 0.08573404, 0.08140642, 0.08051376, 0.08128327,
           0.0776275 , 0.64889558, 0.08165376, 0.08360166, 0.08545804,
           0.08484066, 0.09212734, 0.97408825, 0.08315913, 0.08078309,
           0.94296996, 0.08513671, 0.11524851, 0.08215471, 0.65353923,
           0.11437069, 0.08604866, 0.08211962, 0.09047273, 0.08728701,
           0.09139122, 0.08291739, 0.07885726, 0.08434964, 0.08427168,
           0.08201001, 0.21971799, 0.08478439, 0.08162984, 0.89247909,
           0.08374503, 0.08133641, 0.61901073, 0.08169183, 0.68497934,
           0.08154882, 0.07900925, 0.19706375, 0.0847995 , 0.0828979 ,
           0.08477718, 0.08359476, 0.09364586, 0.08363628, 0.08479827,
           0.08758033, 0.10279445, 0.08745295, 0.59434165, 0.08138641,
           0.30621669, 0.0966963 , 0.08125786, 0.0843131 , 0.08508235,
           0.08289193, 0.10240902, 0.08694654, 0.08418064, 0.5068718 ,
           0.08560549, 0.08002579, 0.09199607, 0.08232822, 0.08838264,
           0.08126839, 0.08369149, 0.14765384, 0.55900556, 0.09612171,
           0.27492562, 0.0790238 , 0.08334724, 0.08594098, 0.08693418,
           0.087929  , 0.08321851, 0.08691786, 0.15604546, 0.08072895,
           0.08065153, 0.082468  , 0.08460935, 0.08629011, 0.0811081 ,
           0.08697047, 0.08434839, 0.08076368, 0.0847092 , 0.09166602,
           0.08686773, 0.08349277, 0.08365693, 0.08678726, 0.08883466,
           0.08054716, 0.08066587, 0.08383334, 0.08973186, 0.08387303,
           0.93350974, 0.07872822, 0.08161481, 0.43641193, 0.08320997,
           0.08092981, 0.0854073 , 0.08323054, 0.19042915, 0.08249607,
           0.0827018 , 0.79905044, 0.08817948, 0.08092733, 0.09206405,
           0.08681285, 0.08492488, 0.07947166, 0.07992544, 0.08340399,
           0.53987275, 0.15872153, 0.11272082, 0.08468026, 0.08169237,
           0.07724638, 0.08698756, 0.08273902, 0.08237868, 0.08719416,
           0.08486132, 0.08245345, 0.08339183, 0.08661901, 0.61481616,
           0.08235798, 0.08077149, 0.07856026, 0.07893729, 0.08074939,
           0.08244256, 0.08969852, 0.0827273 , 0.08029616, 0.08361862,
           0.08177459, 0.0839101 , 0.32505682, 0.17697168, 0.08587896,
           0.08403819, 0.08377523, 0.08232018, 0.08405742, 0.08139792,
           0.08186932, 0.08583442, 0.54297655, 0.08256336, 0.08478341,
           0.08393052, 0.08294815, 0.12991402, 0.08342394, 0.08061944,
           0.73275923, 0.22543242, 0.08697018, 0.08362776, 0.0813177 ,
           0.09518678, 0.0824566 , 0.08501492, 0.85987353, 0.09382662,
           0.79502239, 0.0951015 , 0.0941867 , 0.08152221, 0.0900235 ,
           0.08254301, 0.08329625, 0.10040322, 0.09390556, 0.08125336,
           0.08410294, 0.16394299, 0.08582411, 0.08250436, 0.08418717,
           0.08194805, 0.11822343, 0.07709362, 0.08211582, 0.08859582,
           0.72298196, 0.08552093, 0.08821684, 0.08066418, 0.08320562,
           0.08440588, 0.07972342, 0.08518597, 0.08518285, 0.08856151,
           0.08203569, 0.08355051, 0.08638793, 0.08668831, 0.10237582,
           0.0819398 , 0.09363063, 0.08284859, 0.0874622 , 0.08435574,
           0.75272631, 0.086966  , 0.08248556, 0.08024886, 0.08553947,
           1.01788067, 0.08303262, 0.08983814, 0.08255483, 0.08150664,
           0.08720194, 0.08142342, 0.1196083 , 0.08323487, 0.08261331,
           0.08478155, 0.09467988, 0.08259248, 0.55986422, 0.08045177,
           0.09057627, 0.0859193 , 0.08239075, 0.08420515, 0.09106344,
           0.08038218, 0.08196074, 0.08096948, 0.3972563 , 0.88853347,
           0.08165393, 0.08291527, 0.08336556, 0.08607432, 0.08519584,
           0.08297069, 0.0810091 , 0.08585046, 0.08120228, 0.0982884 ,
           0.10744741, 0.08628468, 0.0851383 , 0.08408805, 0.73433664,
           0.08518511, 0.24684959, 0.0833663 , 0.08301225, 0.0927984 ,
           0.44116245, 0.08762614, 0.08260823, 0.08158648, 0.08030903,
           0.08726111, 0.8032131 , 0.54345299, 0.15298897, 0.08275886,
           0.08729572, 0.10358625, 0.08290123, 0.0842198 , 0.54024682,
           0.08040066, 0.09285065, 0.07948127, 0.08665175, 0.07935279,
           0.0856959 , 0.08168242, 0.08380173, 0.07958354, 0.08338808,
           0.78289322, 0.38227587, 0.08645594, 0.08339125, 0.85182866,
           0.08527194, 0.08335589, 0.08364811, 0.08401467, 0.08065275,
           0.0877438 , 0.08572017, 0.63641327, 0.0835149 , 0.08252026,
           0.08320624, 0.08315776, 0.08961469, 0.08452384, 0.08552498,
           0.07984996, 0.08408204, 0.08445133, 0.08218133, 0.08774106,
           0.07850734, 1.57095393, 0.07996707, 0.08569951, 0.1141534 ,
           0.0859173 , 0.08168136, 0.08885873, 0.07948627, 0.09187077,
           0.08705674, 0.2325834 , 0.08332333, 0.08458558, 0.08233465,
           0.08026828, 0.18950943, 0.08976127, 0.08435681, 0.08497298,
           0.08493242, 0.08230986, 0.21861478, 0.08216878, 0.08619354,
           0.08307133, 0.39339352, 0.08744505, 0.73959515, 0.08618163,
           0.08161971, 0.07856366, 0.08351388, 0.10073845, 0.08677481,
           0.0844343 , 0.44088285, 0.08490545, 0.08090528, 0.08721138,
           0.08483882, 0.08626665, 0.09907192, 0.08271017, 0.08527914,
           0.0933525 , 0.08625187, 0.81187748, 0.08219386, 0.08353065,
           0.08067699, 0.0826846 , 0.08138852, 0.08595854, 0.92019335,
           0.08118303, 0.22016005, 0.07916215, 0.08086794, 0.1012923 ,
           0.08042561, 0.16466937, 0.08463941, 0.08550099, 0.79847527,
           0.09202636, 0.0857501 , 0.08595484, 0.09393829, 0.08086123,
           0.08351929, 0.08964262, 0.08084632, 0.08120662, 0.0820821 ,
           0.08887702, 0.08651905, 0.08109467, 0.08085621, 0.08475753,
           0.09711639, 0.09739515, 0.08744934, 0.0816858 , 0.23977089,
           0.08706071, 0.20027074, 0.08848897, 0.0876273 , 0.08143091,
           0.08227385, 0.08068057, 0.08394448, 0.1267433 , 0.10652113,
           0.9188175 , 0.08911072, 0.08862144, 0.08725366, 0.07689723,
           0.07895012, 0.08329665, 0.08601355, 0.10650172, 0.08587632,
           0.07721861, 0.08611772])

        self.register_buffer("latent_mean", self.latent_mean_orig)
        self.register_buffer("latent_std", self.latent_std_orig)

    def forward(self, x, norm_and_undo=False):
        if norm_and_undo:
            x = (x - self.latent_mean) / self.latent_std
        res = self.transformer_model(x)
        if norm_and_undo:
            res = (res * self.latent_std)
        return res

    def get_input_and_target(self, x):
        sequence_length = x.shape[1]
        input_data = x[:, :(sequence_length - 1)]
        target_data = x[:, 1:(sequence_length)]

        return input_data, target_data

    def loss_function(self, predicted_y, actual_y):
        #v_dot = (predicted_y * actual_y).sum(dim=-1)
        #pred_vec_norm = torch.linalg.vector_norm(predicted_y, dim=-1)
        #actual_vec_norm = torch.linalg.vector_norm(actual_y, dim=-1)
        #eps = 0.00001
        #per_sample_similarity = (v_dot / ((pred_vec_norm * actual_vec_norm) + eps))

        #return (torch.mean(per_sample_similarity) * -1) + F.mse_loss(predicted_y, actual_y)
        return F.mse_loss(predicted_y, actual_y)

    def normalize(self, x):
        return (x - self.latent_mean) / self.latent_std

    def get_loss(self, latents, batch_idx):
        input, target = self.get_input_and_target(latents)
        input_norm = self.normalize(input)
        target_norm = self.normalize(target)
        predicted_transitions_for_target = self(input_norm)
        target_transitions = target_norm - input_norm

        return self.loss_function(predicted_transitions_for_target, target_transitions)

    def training_step(self, batch, batch_idx):
        latents, _ = batch
        loss = self.get_loss(latents, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        latents, _ = batch
        loss = self.get_loss(latents, batch_idx)
        self.log("val/loss", loss)
        if ((batch_idx == 0) and (self.global_rank == 0)):
            self.val_batch = batch
        return

    def log_video(self, tb_logger, name, videos):
        if (self.global_rank != 0):
            return
        for i in range(videos.shape[1]):
            sub_videos = videos[:, i]
            print("original shape ", sub_videos.shape, sub_videos.reshape(videos.shape[0], 3, 256, 256).shape)
            sub_videos = sub_videos.reshape(videos.shape[0], 3, 256, 256)
            tb_logger.add_images(name, videos[:, i], i)

    def decode_64x64(self, images):
        pieces = 8
        split_images = torch.split(images, pieces)
        res = []
        for entry in split_images:
            res.append(self.diffusion_model.decode_encoding(entry))
        return torch.cat(res, dim=0)

    def split_data(self, source, blocks, index):
        new_source = torch.chunk(source, blocks)
        result = new_source[index]
        if (new_source[0].shape[0] != result.shape[0]):
            new_result = torch.zeros_like(new_source[0])
            new_result[:result.shape[0]] = result
            result = new_result
        return result

    def merge_data(self, data, total_processes, expected_seq_size):
        output_list = [torch.empty_like(data) for _ in range(total_processes)]
        torch.distributed.all_gather(output_list, data)
        concated = torch.cat(output_list)
        print(f"concated size: {concated.shape}")
        sys.stdout.flush()
        concated = concated[:expected_seq_size]
        print(f"concated shortened size: {concated.shape}")
        sys.stdout.flush()
        return concated

    def update_val_batch(self, expected_videos=1):
        print(f"Getting update from rank {self.global_rank}")
        sys.stdout.flush()
        if (self.global_rank == 0):
            torch.distributed.broadcast(self.val_batch[0], 0)
            torch.distributed.broadcast(self.val_batch[1], 0)
        else:
            #if (self.val_batch is not None):
            #    del self.val_batch[0]
            #    del self.val_batch[1]
            latents = torch.empty((1,80,512)).to(self.device)
            images = torch.empty((1,80,6,64,64)).to(self.device)
            torch.distributed.broadcast(latents, 0)
            torch.distributed.broadcast(images, 0)
            self.val_batch = [latents, images]
        print(f"Rank: {self.global_rank}, Val_batch: {self.val_batch}")
        sys.stdout.flush()

    def on_validation_epoch_end(self):
        total_nodes = 6
        gpus_per_node = 2
        total_processes = total_nodes * gpus_per_node
        print(f"global_rank {self.global_rank} and total processes {total_processes}")
        sys.stdout.flush()

        if ((self.current_epoch % 50) != 15):
            return

        # Make sure everyone is using the same validation batch
        self.update_val_batch()
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        latents, images = self.val_batch
        number_of_videos = latents.shape[0]
        sequence_length = latents.shape[1]
        input_video_latents, target_video_latents = self.get_input_and_target(latents)
        input_video_images, target_video_images = self.get_input_and_target(images)

        print("Validation - starting data ready!")
        sys.stdout.flush()

        target_video_images = self.diffusion_model.fetch_encoding(target_video_images.reshape(-1, 6, 64, 64), False)
        target_video_original = self.decode_64x64(target_video_images)
        target_video_original = target_video_original.reshape(number_of_videos, sequence_length - 1, 3, 256, 256)
        self.log_video(tb_logger, f"val/original_target_original_{self.current_epoch}", target_video_original)
        print("Target video original done!")
        sys.stdout.flush()

        input_video_latents = input_video_latents.reshape(number_of_videos, sequence_length - 1, 512)
        seed_input_video_latents = input_video_latents[:, 0:1]
        print("seed input initial shape ", seed_input_video_latents.shape)
        for i in range(sequence_length - 1):
            current_prediction = self(seed_input_video_latents, True)
            print(f"current_prediction shape {current_prediction.shape}")
            predicted_next_frame_transition = current_prediction[:, i:(i + 1)]
            predicted_next_frame = seed_input_video_latents[:, i:(i + 1)] + predicted_next_frame_transition
            print("predicted frame shape ", predicted_next_frame.shape)
            seed_input_video_latents = torch.cat([seed_input_video_latents, predicted_next_frame], dim=1)
            print("seed input current shape ", seed_input_video_latents.shape)
        seed_input_video_latents = seed_input_video_latents[:, 1:]
        print("seed input final shape ", seed_input_video_latents.shape)
        target_video_hallucinated = seed_input_video_latents.reshape(-1, 512)
        target_video_hallucinated_split = self.split_data(target_video_hallucinated, total_processes, self.global_rank)
        print("target_video_hallucinated_split shape ", target_video_hallucinated_split.shape)
        x_t_smaller = torch.randn((target_video_hallucinated_split.shape[0],3,64,64)).to(self.device)
        target_video_hallucinated_reconstructed = self.decode_64x64(self.diffusion_model.denoise_zsem(target_video_hallucinated_split, x_t_smaller))
        target_video_hallucinated_reconstructed = self.merge_data(target_video_hallucinated_reconstructed, total_processes, sequence_length - 1)
        target_video_hallucinated_reconstructed = target_video_hallucinated_reconstructed.reshape(number_of_videos, sequence_length - 1, 3, 256, 256)
        self.log_video(tb_logger, f"val/hallucinated_target_{self.current_epoch}", target_video_hallucinated_reconstructed)
        print("Target video hallucinated reconstruction done!")

        target_video_latents = target_video_latents.reshape(-1, 512)
        target_video_latents = self.split_data(target_video_latents, total_processes, self.global_rank)
        target_video_reconstructed = self.decode_64x64(self.diffusion_model.denoise_zsem(target_video_latents, x_t_smaller))
        target_video_reconstructed = self.merge_data(target_video_reconstructed, total_processes, sequence_length - 1)
        target_video_reconstructed = target_video_reconstructed.reshape(number_of_videos, sequence_length - 1, 3, 256, 256)
        print("Target video reconstruction done!")
        self.log_video(tb_logger, f"val/original_target_reconstruction_{self.current_epoch}", target_video_reconstructed)
        sys.stdout.flush()

        input_video_latents = input_video_latents.reshape(number_of_videos, sequence_length - 1, 512)
        target_video_predicted = (input_video_latents + self(input_video_latents, True)).reshape(-1, 512)
        target_video_predicted_split = self.split_data(target_video_predicted, total_processes, self.global_rank)
        target_video_predicted_reconstructed = self.decode_64x64(self.diffusion_model.denoise_zsem(target_video_predicted_split, x_t_smaller))
        target_video_predicted_reconstructed = self.merge_data(target_video_predicted_reconstructed, total_processes, sequence_length - 1)
        target_video_predicted_reconstructed = target_video_predicted_reconstructed.reshape(number_of_videos, sequence_length - 1, 3, 256, 256)
        self.log_video(tb_logger, f"val/predicted_target_{self.current_epoch}", target_video_predicted_reconstructed)
        print("Target video predicted reconstruction done!")

        sys.stdout.flush()

        print("Images added!")
        sys.stdout.flush()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.transformer_model.parameters()), lr=self.config["model"]["base_learning_rate"])
        return optimizer
