import pandas as pd
import numpy as np
# math23k
# bertmtdnn_normalized = [10706, 20255, 1180, 16881, 19972, 16916, 16914, 10516, 10352, 1724, 1721, 14302, 1442, 23038, 22343, 19137, 5721, 4046, 14810, 319, 10912, 7122, 598, 13905, 13902, 19060, 16997, 9116, 21156, 10662, 13485, 20351, 1342, 1345, 6889, 4273, 6125, 12972, 7081, 7090, 19160, 16819, 12124, 5891, 1603, 5472, 10991, 2216, 9287, 2990, 3792, 3247, 6531, 7370, 7364, 11935, 11067, 3852, 15218, 21054, 18375, 884, 883, 882, 1265, 9823, 5370, 9600, 6604, 5958, 12375, 2910, 20199, 17818, 17815, 2314, 19579, 22274, 4285, 19780, 20729, 13294, 20093, 7210, 14449, 17407, 11809, 12868, 16411, 9495, 9498, 21321, 22983, 5002, 5001, 7179, 14767, 11278, 17301, 17310, 13109, 22411, 15166, 4205, 4586, 4583, 11183, 7707, 19187, 10450, 2832, 1433, 1432, 1440, 9895, 22598, 4877, 18472, 10135, 2270, 11090, 19020, 17724, 3237, 12799, 548, 546, 19604, 17525, 13459, 11035, 3336, 7037, 15485, 11337, 11339, 16596, 12683, 1924, 20028, 1216, 18910, 3110, 5137, 10274, 10272, 2980, 4934, 2109, 2107, 11878, 16033, 19337, 5301, 19526, 1890, 3574, 16512, 4705, 7894, 1748, 1741, 2682, 14477, 5611, 10185, 1159, 22658, 4199, 7146, 18748, 17377, 16688, 17086, 20599, 1589, 20334, 9925, 15155, 7758, 19206, 19207, 11457, 10477, 8921, 1465, 15691, 20148, 6338, 11856, 11857, 5411, 334, 10937, 20423, 16999, 21159, 21154, 3979, 572, 576, 5879, 16609, 10609, 15417, 9637, 3350, 3347, 7062, 11370, 1978, 20916, 20912, 13894, 2232, 22191, 15791, 4141, 17554, 1060, 11047, 12532, 16326, 19519, 1876, 17060, 355, 15181, 15189, 12183, 17696, 14913, 4680, 7224, 15881, 18901, 8213, 11551, 268, 17250, 21598, 21593, 3775, 8026, 7447, 7450, 11743, 11750, 9189, 12215, 8853, 8855, 16202, 9974, 18555, 19754, 17613, 20705, 10341, 1713, 19258, 16210, 17154, 9391, 9399, 16774, 13332, 16432, 16435, 22970, 3942, 22146, 18794, 12741, 17321, 17327, 16990, 9142, 13165, 15909, 20362, 1083, 8695, 8698, 8694, 21801, 3396, 3395, 3399, 16281, 12159, 18460, 14135, 15663, 4040, 6370, 6367, 6365, 14898, 14893, 5199, 14274, 5443, 7539, 19360, 17300, 17293, 18174, 11864, 6508, 4172, 20678, 19346, 7311, 21263, 21264, 1828, 3311, 3314, 636, 3867, 11649, 753, 15202, 16580, 19871, 14965, 20003, 1277, 19726, 19724, 2594, 20948, 4690, 15507, 5948, 12370, 2912, 21979, 2124, 4915, 12432, 18250, 21098, 2156, 19543, 22482, 20731, 13283, 20084, 4609, 13861, 5675, 21655, 13095, 20293, 2709, 3919, 6736, 22173, 6087, 7169, 8072, 19015, 5850, 5849, 17900, 12797, 18530, 879, 871, 4592, 4598, 11191, 19192, 13778, 17104, 10453, 1407, 1405, 1542, 6314, 6318, 10126, 1649, 5439, 8584, 2260, 17734, 9796, 20402, 20409, 551, 18144, 20641, 7324, 15904, 1385, 9618, 21254, 3324, 3817, 22724, 5281, 5286, 12933, 12938, 14692, 19111, 19113, 19118, 16581, 12697, 13259, 1484, 20060, 14567, 1222, 8479, 2096, 2095, 2099, 18409, 4380, 7091, 5916, 5914, 17225, 21684, 21531, 1707, 5261, 4941, 4944, 14799, 19677, 6059, 10008, 15988, 1895, 1891, 17071, 20569, 20562, 20564, 1531, 10421, 10428, 18062, 16507, 16506, 17671, 13566, 17169, 9869, 22004, 107, 15870, 18923, 14404, 13836, 13831, 13839, 19382, 21608, 8537, 3754, 9766, 8047, 20876, 9953, 782, 15121, 6263, 14319, 14313, 12416, 7743, 7747, 17138, 8937, 1471, 1477, 22380, 22376, 22089, 184, 14482, 14490, 15682, 15359, 17445, 13355, 22941, 21364, 22127, 18801, 14724, 18116, 1354, 22451, 2477, 3379, 3375, 8169, 19147, 21738, 20062, 19738, 18436, 21015, 21011, 6693, 8286, 7560, 17949, 5929, 11656, 17767, 2147, 2145, 69, 19835, 6067, 10039, 5344, 11077, 15980, 12522, 10806, 2388, 775, 5756, 5760, 15227, 12641, 12650, 9311, 17685, 17689, 10490, 10483, 14905, 1259, 22552, 3147, 15527, 13390, 11891, 20190, 15094, 7927, 100, 17826, 11286, 21426, 8845, 16384, 19574, 14643, 4746, 19767, 817, 1709, 14323, 12444, 12449, 2533, 17170, 4887, 15309, 21674, 13308, 20104, 11816, 15549, 22974, 18785, 19698, 9132, 7924, 4563, 1096, 22430, 21815, 18664, 7714, 16874, 21744, 4802, 18465, 14124, 6497, 3199, 292, 14246, 7698, 16572, 379, 11626, 19878, 3224, 18164, 534, 6516, 21140, 17511, 18208, 7308, 11911, 11007, 10835, 2401, 3877, 22296, 4571, 4573, 4579, 14545, 13795, 11497, 1935, 1209, 9803, 2615, 9025, 4313, 6553, 3118, 3111, 6662, 8255, 15511, 15517, 11515, 9223, 2968, 20208, 18258, 6597, 18470, 19028, 4102, 15992, 4485, 5317, 17875, 19551, 16084, 3509, 4735, 17657, 13547, 14442, 7267, 6205, 11167, 23048, 23046, 22028, 15849, 15843, 14466, 12025, 10772, 10778, 11440, 10195, 1196, 3016, 6750, 5398, 5396, 8062, 7482, 5834, 13128, 13123, 21847, 3587, 15146, 11627, 15422, 19218, 13749, 17118, 17117, 17111, 10469, 8913, 22313, 8156, 19982, 20179, 5403, 9474, 17709, 20433, 3213, 16834, 11941, 3356, 3822, 7059, 7057, 8149, 19296, 17198, 17193, 8998, 6511, 6517, 1500, 9136, 4348, 7578, 13690, 11100, 16787, 21529, 4952, 3300, 5364, 7436, 13634, 2369, 17068, 21024, 12194, 9872, 995, 1000, 20986, 6635, 4643, 5152, 5635, 5638]
# bertmtdnn_ansids = [16384, 18436, 4102, 8213, 22552, 16411, 18460, 10272, 18465, 10274, 18470, 18472, 4141, 16432, 2096, 16435, 2099, 2107, 2109, 6205, 8255, 14404, 69, 22598, 4172, 2124, 12370, 12375, 20569, 8286, 2145, 18530, 2147, 100, 10341, 4199, 14442, 2156, 4205, 107, 10352, 14449, 20599, 6263, 16506, 18555, 16507, 16512, 12416, 22658, 14466, 14477, 12432, 14482, 14490, 12444, 20641, 12449, 2216, 6314, 6318, 10421, 2232, 184, 4285, 10428, 16572, 6338, 16580, 22724, 20678, 16581, 10450, 10453, 2260, 14545, 4313, 6365, 2270, 16609, 20705, 6370, 10469, 14567, 18664, 10477, 10483, 12532, 20729, 20731, 10490, 4348, 2314, 268, 10516, 4380, 8479, 292, 16688, 14643, 18748, 319, 2369, 334, 2388, 8537, 12641, 18785, 355, 14692, 6497, 2401, 18794, 12650, 6508, 6511, 10609, 18801, 6516, 6517, 6531, 14724, 16774, 8584, 4485, 12683, 20876, 16787, 12697, 6553, 22941, 10662, 2477, 14767, 20912, 16819, 20916, 22974, 16834, 12741, 6597, 22983, 6604, 14799, 10706, 4563, 20948, 18901, 14810, 4571, 4573, 18910, 4579, 2533, 4583, 4586, 18923, 16874, 6635, 4592, 16881, 8694, 8695, 4598, 8698, 20986, 12797, 23038, 12799, 4609, 6662, 23048, 23046, 21011, 16916, 10772, 534, 21015, 5398, 10778, 546, 2594, 548, 6693, 551, 21024, 4643, 14893, 14898, 10806, 2615, 14905, 572, 21054, 576, 14913, 19015, 4680, 19020, 6736, 4690, 10835, 19028, 598, 16990, 6750, 4705, 16997, 16999, 21098, 19060, 14965, 2682, 636, 4735, 12933, 12938, 4746, 8845, 21140, 8853, 2709, 10912, 21154, 21156, 17060, 21159, 19111, 19113, 17068, 19118, 17071, 10937, 17086, 19137, 4802, 19147, 17104, 8913, 17111, 19160, 8921, 17117, 17118, 6889, 8937, 10991, 753, 17138, 19187, 15094, 19192, 11007, 17154, 19206, 19207, 21254, 775, 4877, 782, 21263, 2832, 21264, 17169, 15121, 17170, 19218, 4887, 11035, 8998, 11047, 13095, 17193, 15146, 17198, 817, 15155, 4915, 13109, 19258, 11067, 15166, 9025, 13123, 11077, 4934, 13128, 21321, 17225, 15181, 4941, 4944, 11090, 15189, 4952, 11100, 2910, 2912, 17250, 15202, 19296, 13165, 879, 15218, 883, 884, 882, 21364, 15227, 7037, 5001, 5002, 19337, 17293, 7057, 19346, 7059, 17300, 17301, 7062, 2968, 9116, 17310, 11167, 19360, 2980, 7081, 17321, 9132, 2990, 11183, 17327, 9136, 7090, 7091, 21426, 9142, 11191, 19382, 3016, 13259, 15309, 7122, 17377, 13283, 9189, 995, 1000, 7146, 13294, 13634, 13308, 17407, 15359, 7169, 9223, 7179, 11278, 5137, 13332, 11286, 21529, 5152, 1060, 17445, 3110, 3111, 7210, 13355, 3118, 15682, 7224, 15417, 1083, 15422, 19519, 19526, 9287, 11337, 1096, 11339, 3147, 13390, 5199, 19543, 21593, 21598, 9311, 19551, 7267, 17511, 21608, 11370, 17525, 19574, 19579, 15485, 3199, 1159, 7308, 5261, 3213, 7311, 17554, 13459, 19604, 15507, 21655, 3224, 15511, 1180, 7324, 15517, 5281, 3237, 5286, 15527, 21674, 1196, 13485, 3247, 9391, 11440, 21684, 5301, 9399, 1209, 15549, 1216, 11457, 7364, 5317, 1222, 7370, 17613, 19677, 5344, 3300, 21738, 1259, 11497, 13547, 3311, 21744, 1265, 3314, 19698, 5364, 17657, 5370, 11515, 3324, 1277, 13566, 17671, 3336, 19724, 7436, 19726, 3347, 5396, 17685, 3350, 9495, 7447, 17689, 9498, 7450, 19738, 11551, 17696, 5403, 3356, 5411, 21801, 19754, 17709, 15663, 3375, 3379, 19767, 21815, 7482, 17724, 1342, 5439, 1345, 3395, 19780, 3396, 3399, 5443, 15691, 17734, 1354, 21847, 5472, 17767, 1385, 11626, 11627, 7539, 13690, 19835, 1405, 1407, 9600, 11649, 7560, 11656, 9618, 17815, 1432, 1433, 17818, 7578, 19871, 1440, 1442, 17826, 9637, 19878, 15791, 3509, 13749, 1465, 1471, 1477, 1484, 13778, 17875, 21979, 1500, 13795, 15843, 11750, 15849, 5611, 17900, 22004, 3574, 1531, 15870, 3587, 19972, 5635, 1542, 13831, 15881, 5638, 13836, 22028, 19982, 13839, 7698, 7707, 17949, 15904, 11809, 20003, 7714, 15909, 13861, 9766, 11816, 5675, 1589, 20028, 7743, 1603, 9796, 13894, 7747, 22089, 9803, 13902, 7758, 11856, 13905, 11857, 11864, 5721, 20060, 20062, 9823, 11878, 15980, 22127, 11891, 20084, 15988, 15992, 5756, 20093, 5760, 22146, 11911, 20104, 9869, 18062, 9872, 22173, 11935, 16033, 9895, 3754, 1707, 1709, 22191, 1713, 20148, 1721, 1724, 3775, 18116, 9925, 5834, 1741, 3792, 20179, 1748, 16084, 7894, 5849, 5850, 20190, 18144, 9953, 20199, 3817, 3822, 20208, 7924, 18164, 9974, 5879, 7927, 12025, 18174, 22274, 5891, 3852, 10008, 22296, 3867, 5916, 5914, 20255, 18208, 1828, 3877, 5929, 22313, 14124, 14135, 10039, 5948, 20293, 5958, 22343, 16202, 18250, 3919, 16210, 18258, 1876, 12124, 1890, 1891, 1895, 22376, 22380, 20334, 8047, 8062, 20351, 12159, 1924, 8072, 20362, 22411, 3979, 10126, 1935, 10135, 12183, 16281, 22430, 12194, 14246, 6059, 20402, 22451, 6067, 12215, 20409, 1978, 14274, 16326, 18375, 20423, 10185, 4040, 6087, 4046, 20433, 22482, 10195, 8149, 8156, 14302, 14313, 8169, 14319, 14323]
# mawps

bertmtdnn_normalized = [245, 2489, 3607, 3415, 44, 146, 3711, 2639, 2307, 3774, 2269, 2356, 3874, 67, 3629, 2830, 125, 3402, 3314, 2947, 177, 2814, 3196, 3139, 3529, 3150, 3336, 285, 2554, 3859, 3720, 3137, 18, 3772, 2673, 3504, 2448, 3381, 3177, 3233, 3650, 293, 2277, 2695, 3175, 3587, 61, 3140, 2876, 2508, 3584, 2406, 3890, 1041, 2532, 2299, 3047, 3067, 3107, 535, 2993, 284, 2831, 3616, 116, 3247, 987, 2395, 3766, 3734, 2622, 212, 2474, 2869, 3512, 2764, 2466, 176, 3130, 122, 3727, 2878, 2933, 211, 3416, 2399, 2690, 3740, 2407, 2594, 2605, 200, 3350, 2439, 75, 2666, 2746, 2284, 197, 3672, 2256, 3420, 3122, 3549, 3121, 3014, 3207, 2607, 2644, 2899, 3433, 3234, 2700, 3456, 2794, 3751, 2578, 3557, 2580, 2811, 2884, 2990, 3572, 3403, 3653, 191, 156, 129, 3249, 2930, 2371, 3297, 3086, 2659, 3682, 3358, 2910, 3679, 3750, 3182, 2264, 2303, 278, 3520, 266, 338, 3466, 3322, 2621, 251, 3269, 2927, 3482, 3748, 350, 347, 2358, 79, 3725, 2701, 3114, 3132, 128, 3309, 2498, 3289, 2890, 3304, 2635, 3862, 2272, 2740, 2702, 3189, 3613, 2821, 2546, 3757, 2261, 3827, 3363, 3025, 2560, 2692, 3286, 313, 105, 2278, 2939, 2640, 195, 3480, 835, 3747, 2958, 3069, 3718, 2941, 2913, 3300, 3722, 3435, 16, 2736, 3585, 3783, 2419, 2777, 3502, 8, 34, 3260, 360, 2649, 140, 2432, 2852, 2263, 3419, 2342, 3245, 2987, 3781, 2992, 2313, 2632, 3599, 3344, 3418, 3731, 216, 3311, 3335, 2427, 3870, 3468, 2671, 3186, 3273, 3801, 3675, 3161, 3684, 132, 3550, 3619, 342, 2926, 2963, 2664, 299, 204, 2745, 3061, 3183, 3351, 3810, 3474, 2400, 357, 2771, 2728, 236, 2802, 58, 2911, 2759, 3439, 3562, 3776, 90, 3654, 2862, 2593, 68, 2717, 354, 2905, 3406, 394, 41, 807, 2841, 2537, 3643, 918, 2512, 2290, 2684, 2694, 3812, 2837, 258, 100, 2967, 3293, 3705, 2348, 203, 2405, 2398, 2971, 2710, 356, 2653, 2316, 3568, 319, 2765, 2847, 2360, 2552, 135, 2318, 3846, 2922, 184, 2781, 263, 3033, 3712, 2511, 3805, 3181, 3228, 179, 3515, 3280, 2795, 3399, 2629, 3898, 382, 93, 2444, 3597, 2535, 76, 2663, 2704, 2843, 2366, 2834, 2314, 362, 2817, 2485, 3652, 2577, 2247, 2654, 39, 3818, 986, 3894, 2543, 3824, 155, 3427, 2931, 1690, 3266]
bertmtdnn_ansids = [8, 16, 18, 34, 39, 41, 44, 58, 61, 67, 68, 75, 76, 79, 90, 93, 100, 105, 116, 122, 125, 128, 129, 132, 135, 140, 155, 156, 176, 177, 179, 184, 191, 195, 197, 2247, 200, 203, 204, 2256, 211, 212, 2261, 2263, 2264, 216, 2269, 2272, 2277, 2278, 2284, 236, 2290, 245, 2299, 251, 2303, 2307, 258, 263, 2313, 266, 2314, 2316, 2318, 278, 284, 285, 293, 2342, 299, 2348, 2356, 2358, 2360, 313, 2366, 319, 2371, 338, 342, 2395, 347, 350, 2399, 2400, 2398, 354, 356, 357, 2406, 2407, 360, 2405, 362, 2419, 2427, 382, 2432, 2439, 394, 2444, 2448, 2466, 2474, 2485, 2489, 2498, 2508, 2511, 2512, 2532, 2535, 2537, 2543, 2546, 2552, 2554, 2560, 2577, 2578, 2580, 535, 2593, 2594, 2605, 2607, 2621, 2622, 2629, 2632, 2635, 2639, 2640, 2644, 2649, 2653, 2654, 2659, 2663, 2664, 2666, 2671, 2673, 2684, 2690, 2692, 2694, 2695, 2700, 2701, 2702, 2704, 2710, 2717, 2728, 2736, 2740, 2745, 2746, 2759, 2764, 2765, 2771, 2777, 2781, 2794, 2795, 2802, 2811, 2814, 2817, 2821, 2830, 2831, 2834, 2837, 2841, 2843, 2847, 2852, 807, 2862, 2869, 2876, 2878, 835, 2884, 2890, 2899, 2905, 2910, 2911, 2913, 2922, 2926, 2927, 2930, 2931, 2933, 2939, 2941, 2947, 2958, 2963, 2967, 2971, 2987, 2990, 2992, 2993, 3014, 3025, 3033, 987, 3047, 3061, 3067, 3069, 3086, 1041, 3114, 3121, 3122, 3130, 3132, 3137, 3139, 3140, 3150, 3161, 3175, 3177, 3181, 3182, 3183, 3186, 3189, 3196, 3207, 3228, 3233, 3234, 3247, 3249, 3260, 3266, 3269, 3273, 3280, 3286, 3289, 3293, 3297, 3300, 3304, 3309, 3311, 3314, 3322, 3335, 3336, 3344, 3351, 3358, 3363, 3381, 3399, 3402, 3403, 3406, 3415, 3416, 3418, 3419, 3420, 3427, 3433, 3435, 3439, 3456, 3466, 3468, 3474, 3480, 3482, 3502, 3504, 3512, 3515, 3520, 3529, 3549, 3550, 3557, 3562, 3568, 3572, 3584, 3585, 3587, 3597, 3599, 3607, 3613, 3616, 3619, 3629, 3643, 3650, 3652, 3653, 3654, 3672, 3675, 3679, 3682, 3684, 3705, 3711, 3712, 3718, 3722, 3725, 3727, 3731, 3734, 1690, 3740, 3747, 3748, 3750, 3751, 3766, 3772, 3774, 3776, 3781, 3783, 3801, 3810, 3812, 3818, 3824, 3827, 3846, 3859, 3862, 3870, 3890, 3894, 3898]


title = ['math23k_cls','math23k_gene','mawps_cls','mawps_gene']
data_path = ["/home/dg/PycharmProjects/Deep-Reinforcement-Learning-Hands-On/Chapter18/dataset/Math23k/tokens/{}.tsv",
             "/home/dg/PycharmProjects/Deep-Reinforcement-Learning-Hands-On/Chapter18/dataset/mawps/fold{}/tokens/{}.tsv",
             "/home/dg/PycharmProjects/Deep-Reinforcement-Learning-Hands-On/Chapter18/dataset/oalg514/fold{}/tokens/{}.tsv"]
MAWPS_BERT_CLS = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-24-16:21:31_mawps_0_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-3689/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-24-19:23:45_mawps_1_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-2499/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-24-21:36:58_mawps_2_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-10234/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-24-23:49:56_mawps_3_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-13804/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-25-02:02:55_mawps_4_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-4284/BestBertAbsSum.csv']
MAWPS_ELECTRA_CLS = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-26-09:00:48_mawps_0_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-14994/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-25-15:55:24_mawps_1_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-14994/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-25-18:08:55_mawps_2_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-6664/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-25-20:21:56_mawps_3_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-16779/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-25-22:35:22_mawps_4_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-13209/BestBertAbsSum.csv'
                ]
MAWPS_BERT_GENE = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/mawps/model_08-14-21:44:40_mawps_0_16_150_bert_5e-5_2e-4_bert-base-uncased_base/checkpoint-11424/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/mawps/model_08-15-03:18:17_mawps_1_16_150_bert_5e-5_2e-4_bert-base-uncased_base/checkpoint-11424/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/colab/result/mawps/model_08-15-00_25_33_mawps_2_16_150_bert_5e-5_2e-4_bert-base-uncased_MTLgene/checkpoint-10829/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/colab/result/mawps/model_08-15-04_13_13_mawps_3_16_150_bert_5e-5_2e-4_bert-base-uncased_MTLgene/checkpoint-12614/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/colab/result/mawps/model_08-15-08_00_27_mawps_4_16_150_bert_5e-5_2e-4_bert-base-uncased_MTLgene/checkpoint-13804/BestBertAbsSum.csv'
                ]
MAWPS_ELECTRA_GENE = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/result/colab/result/mawps/model_08-15-11_52_09_mawps_0_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTLgene/checkpoint-16779/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/mawps/model_08-15-09:37:07_mawps_1_16_150_electra_0.5e-4_2e-4_google/electra-base-discriminator_base/checkpoint-9639/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/mawps/model_08-15-15:11:00_mawps_2_16_150_electra_0.5e-4_2e-4_google/electra-base-discriminator_base/checkpoint-9639/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/colab/result/mawps/model_08-15-15_40_44_mawps_3_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTLgene/checkpoint-13209/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/colab/result/mawps/model_08-15-19_28_15_mawps_4_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTLgene/checkpoint-12614/BestBertAbsSum.csv'
                ]
MAWPS_ELECTRA_GENE_BASE = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Coling/MAWPS/ablation/base/model_06-24-00:44:50_mawps_0_32_115_electra_1e-4_2e-4_google/electra-base-discriminator_-opff_aux/checkpoint-4860/BestBertAbsSum.csv',
                ]

MAWPS_ELECTRA_GENE_withoutLM = [
                '/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/mawps/model_01-04-19:22:47_mawps_0_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-15589/BestBertAbsSum.csv',
                '/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/mawps/model_01-04-21:12:15_mawps_1_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-14399/BestBertAbsSum.csv',
                '/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/mawps/model_01-04-23:01:35_mawps_2_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-10829/BestBertAbsSum.csv',
                '/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/mawps/model_01-05-00:51:04_mawps_3_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-8449/BestBertAbsSum.csv',
                '/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/mawps/model_01-05-02:40:20_mawps_4_16_150_electra_5e-5_2e-4_google/electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-17969/BestBertAbsSum.csv'
                ]


Math23k_BERT_CLS = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/Math23k/model_08-24-20:57:46_Math23k_1_16_150_bert_5e-5_2e-4_bert-base-chinese_cls/AbsSum150.csv']
Math23k_BERT_GENE = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/quadro/model_08-25-13:22:13_Math23k_1_16_150_bert_5e-5_2e-4_bert-base-chinese_cls/AbsSum150.csv']
Math23k_ELECTRA_CLS = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/Math23k/model_08-25-21:33:42_Math23k_1_16_150_auto_3e-5_7e-5_hfl/chinese-electra-base-discriminator_cls/AbsSum150.csv']

Math23k_ELECTRA_GENE = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/quadro/model_08-27-03:51:29_Math23k_1_16_150_auto_3e-5_7e-5_hfl/chinese-electra-base-discriminator_gene/AbsSum150.csv']
Math23k_ELECTRA_GENE_withoutLM = ['/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/Math23k/model_01-04-19:23:34_Math23k_1_16_150_auto_3e-5_7e-5_hfl/chinese-electra-base-discriminator_MTL_generation_withoutEnPa/checkpoint-209135/BestBertAbsSum.csv']
Math23k_ELECTRA_GENE_base = ['/home/dg/PycharmProjects/Coling_JournalofSuperComputing/savefile/result/Math23k/model_11-19-13:39:46_Math23k_1_16_150_auto_3e-5_7e-5_hfl/chinese-electra-base-discriminator_base/checkpoint-209135/BestBertAbsSum.csv']

alg514_BERT_CLS = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/BERT/model_08-28-12:39:21_alg514_0_16_150_bert_4e-5_7e-5_bert-base-uncased_cls/checkpoint-546/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/BERT/model_08-28-13:28:54_alg514_1_16_150_bert_3e-5_7e-5_bert-base-uncased_cls/checkpoint-1976/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/BERT/model_08-28-14:25:18_alg514_2_16_150_bert_3e-5_7e-5_bert-base-uncased_cls/checkpoint-1846/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/BERT/model_08-28-15:21:40_alg514_3_16_150_bert_3e-5_7e-5_bert-base-uncased_cls/checkpoint-546/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/BERT/model_08-28-16:17:05_alg514_4_16_150_bert_3e-5_7e-5_bert-base-uncased_cls/checkpoint-1066/BestBertAbsSum.csv']
alg514_ELECTRA_CLS = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/ELECTRA/model_08-28-12:49:13_alg514_0_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-546/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/ELECTRA/model_08-28-14:19:23_alg514_1_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-1586/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/ELECTRA/model_08-28-14:48:38_alg514_2_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-1196/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/ELECTRA/model_08-28-15:18:00_alg514_3_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-806/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/cls/ELECTRA/model_08-28-15:47:20_alg514_4_16_150_electra_3e-5_7e-5_google/electra-base-discriminator_cls/checkpoint-1196/BestBertAbsSum.csv'
                ]
alg514_BERT_GENE = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/BERT/model_08-28-12:52:20_alg514_0_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-2756/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/BERT/model_08-28-13:47:13_alg514_1_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-2886/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/BERT/model_08-28-14:39:24_alg514_2_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-2236/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/BERT/model_08-28-15:25:30_alg514_3_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-3406/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/BERT/model_08-28-16:11:55_alg514_4_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-3146/BestBertAbsSum.csv'
                ]
alg514_ELECTRA_GENE = [
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/ELECTRA/model_08-28-16:51:14_alg514_0_16_150_electra_1e-4_2e-4_google/electra-base-discriminator_cls/checkpoint-2626/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/ELECTRA/model_08-28-17:49:05_alg514_1_16_150_electra_1e-4_2e-4_google/electra-base-discriminator_cls/checkpoint-3276/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/ELECTRA/model_08-28-18:46:32_alg514_2_16_150_electra_1e-4_2e-4_google/electra-base-discriminator_cls/checkpoint-1716/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/ELECTRA/model_08-28-19:44:09_alg514_3_16_150_electra_1e-4_2e-4_google/electra-base-discriminator_cls/checkpoint-3536/BestBertAbsSum.csv',
                '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/StateOfTheArt/Supercomputing/oalg514/gene/ELECTRA/model_08-28-20:41:39_alg514_4_16_150_electra_1e-4_2e-4_google/electra-base-discriminator_cls/checkpoint-3276/BestBertAbsSum.csv'
                ]
title = ['math23k_cls','math23k_gene','mawps_cls','mawps_gene','alg514_BERT_CLS','alg514_BERT_GENE']

# cand_path = ['/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/Math23k/model_08-24-20:57:46_Math23k_1_16_150_bert_5e-5_2e-4_bert-base-chinese_cls/AbsSum150.csv',
#     "/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/quadro/model_08-25-13:22:13_Math23k_1_16_150_bert_5e-5_2e-4_bert-base-chinese_cls/AbsSum150.csv",
#              '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/ks/savefile/result/mawps/model_08-24-16:21:31_mawps_0_16_150_bert_5e-5_2e-4_bert-base-uncased_cls/checkpoint-3689/BestBertAbsSum.csv',
#              '/media/dg/4696e64f-cc31-4830-b9ea-368b91c87872/Supercomputing_savefile/result/mawps/model_08-14-21:44:40_mawps_0_16_150_bert_5e-5_2e-4_bert-base-uncased_base/checkpoint-11424/BestBertAbsSum.csv']
# cand_path = [Math23k_BERT_CLS[0],Math23k_BERT_GENE[0],MAWPS_BERT_CLS[0],MAWPS_BERT_GENE[0]]
# cand_path = [Math23k_ELECTRA_CLS[0],Math23k_ELECTRA_GENE[0],MAWPS_ELECTRA_CLS[0],MAWPS_ELECTRA_GENE[0]]
# cand_path = [Math23k_ELECTRA_CLS[0],Math23k_ELECTRA_GENE[0],MAWPS_ELECTRA_CLS,MAWPS_ELECTRA_GENE]
# cand_path = [Math23k_Base_ELECTRA_GENE[0]]
# cand_path = [MAWPS_ELECTRA_GENE_BASE[0]]
cand_path = [Math23k_ELECTRA_GENE_base[0]]
# cand_path = [Math23k_BERT_CLS[0]]
# cand_path = [Math23k_ELECTRA_GENE_withoutLM[0]]
# cand_path = [MAWPS_BERT_CLS[0]]
# cand_path = [MAWPS_BERT_GENE[0]]
valid_num = 5
onlytemplate_acc = False
if valid_num != 1:
    onlytemplate_acc = True

# for pathlist in [Math23k_BERT_CLS,Math23k_ELECTRA_CLS,Math23k_BERT_GENE,Math23k_ELECTRA_GENE]:
for pathlist in [alg514_BERT_CLS,alg514_BERT_GENE,alg514_ELECTRA_CLS,alg514_ELECTRA_GENE]:
    acclist = []
    templateacclist = []
    for valid in range(valid_num):
        path = pathlist[valid]
        result_csv = pd.read_csv(path, sep='\t')
        result_csv['equation matching'] = result_csv.apply(lambda x: (
                x['ref_list'].replace('##', '').replace(' ', '') == x['hyp_list'].replace('##', '').replace(' ', '')),
                                                           axis=1)
        NumofEquationMatching = len(result_csv[result_csv['equation matching'] == True])
        acclist.append(result_csv['equation_accuracy'].values[0])
        templateacclist.append(NumofEquationMatching / len(result_csv))
        print("Equation templat acc. : {}".format(NumofEquationMatching / len(result_csv)))
    print(pathlist)
    print('a : {}'.format(np.mean(acclist)))
    print('ta : {}'.format(np.mean(templateacclist)))
valid_num = 0
for n,path in enumerate(cand_path):
    if len(path) < 10:
        path = path[valid_num]

    print(path + '\n')
    print(data_path[n//2]+'\n')
    result_csv = pd.read_csv(path,sep='\t')
    # result_csv['equation matching'] = result_csv.apply(lambda x: (
    #             x['ref_list'].replace('##', '').replace(' ', '') == x['hyp_list'].replace('##', '').replace(' ', '')),
    #                                                    axis=1)
    # NumofEquationMatching = len(result_csv[result_csv['equation matching']==True])
    #
    # print("Equation templat acc. : {}".format(NumofEquationMatching/len(result_csv)))
    # if onlytemplate_acc:
    #     continue
    result_csv['op_num'] = result_csv['ref_list'].apply(
        lambda x: (x.count('+') + x.count('*') + x.count('-') + x.count('/')))

    result_csv['bertmtdnn_ansids'] = result_csv['idx_list'].apply(lambda x: (x in bertmtdnn_ansids))
    result_csv['bertmtdnn_normalized'] = result_csv['idx_list'].apply(lambda x: (x in bertmtdnn_normalized))

    # equation tmp solving acc
    result_csv['matching'] = result_csv.apply(
        lambda x: x['ref_list'].replace('##', '') ==
                  x['hyp_list'], axis=1)
    #
    for op in range(11):
        cand = result_csv[result_csv['op_num'] == op]
        # print("{} {} {} {} {} {}".format(op,len(cand) ,'/',np.sum(cand['matching'].values),':', np.sum(cand['matching'].values)/len(cand)))
        # print("{} {} {} {} {} {}".format(op,len(cand),'/',np.sum(cand['bertmtdnn_ansids'].values),':',  np.sum(cand['bertmtdnn_ansids'].values)/len(cand)))
    if n//2 >= 1:
        train = pd.read_csv(data_path[n//2].format(valid_num,'train'),
                    sep='\t',names=['index','question','equation','group','answer','numbers','operatordiff'])
        test = pd.read_csv(data_path[n//2].format(valid_num,'test'),
                    sep='\t',names=['index','question','equation','group','answer','numbers','operatordiff'])
    else:
        train = pd.read_csv(data_path[n // 2].format('train'),
                            sep='\t',
                            names=['index', 'question', 'equation', 'group', 'answer', 'numbers', 'operatordiff'])
        test = pd.read_csv(data_path[n // 2].format('test'),
                           sep='\t',
                           names=['index', 'question', 'equation', 'group', 'answer', 'numbers', 'operatordiff'])

    train['constants'] = train['equation'].apply(lambda x: ('C' in x))
    test['constants'] = test['equation'].apply(lambda x: ('C' in x))
    neq = len(list(set(list(train['equation'].values) + list(test['equation'].values))))
    print("The number of equation template : {}".format(neq))

    tcp = len(train[train['constants'] == True])
    tecp = len(test[test['constants'] == True])
    wcp = tcp + tecp
    print("")
    # print("whole constant problem : {}".format(wcp))
    # print("train constant problem : {}".format(tcp))
    # print("test constant problem : {}".format(tecp))
    test['intraing'] = test['equation'].apply(lambda x :(x in train['equation'].values))
    nottrain_indices = test[test['intraing'] == False]['index'].values
    print('unseen equation template:{}'.format(len(list(set(list(test[test['intraing'] == False]['equation'].values))))))

    result_csv['not_intraining'] = result_csv['idx_list'].apply(lambda x:(x in nottrain_indices))
    # print('mtdnn_matched unsenn equation template: {}'.format(len(list(set(list(
    #     result_csv[(result_csv['not_intraining'] == True) & (result_csv['bertmtdnn_ansids'] == True)][
    #         'hyp_list'].values))))))

    print(
        'matched unseen equation template:{}'.
            format(len(list(set(list(result_csv[(result_csv['not_intraining'] == True)&(result_csv['matching'] == True)]['hyp_list'].values))))))

    result_csv['notmatching_op'] = result_csv.apply(lambda x:x['ref_list'].replace('##','').replace('*','op').replace('+','op').replace('-','op').replace('/','op')==x['hyp_list'].replace('*','op').replace('+','op').replace('-','op').replace('/','op'),axis=1)
    misidentificationerror =len(result_csv[(result_csv['notmatching_op']==True)&(result_csv['matching']==False)])

    error = len(result_csv[result_csv['matching']==False])
    print('{} : {}'.format('corr', len(
        result_csv[result_csv['matching'] == True])))
    print('{} : {}'.format('incorr',len(result_csv[(result_csv['notmatching_op']==False)&(result_csv['matching']==False)])))
    print("{} : {}/{}={}".format('misidentifying operator',misidentificationerror,error,misidentificationerror/error))
    # print('{} : {}'.format('corr', len(result_csv[result_csv['bertmtdnn_ansids'] == True])))
    # misidentificationerror_MTDNN = len(result_csv[(result_csv['bertmtdnn_ansids']==False)&(result_csv['bertmtdnn_normalized']==True)])
    # MTDNN_error = len(result_csv[result_csv['bertmtdnn_ansids']==False])


    # print('{} : {}'.format('incorr', len(
    #     result_csv[(result_csv['bertmtdnn_ansids']==False)&(result_csv['bertmtdnn_normalized']==False)])))
    # print("{} : {}/{}={}".format('mtdnn_misidentifying operator', misidentificationerror_MTDNN, MTDNN_error,
    #                              misidentificationerror_MTDNN / MTDNN_error))

    result_csv['constants'] = result_csv['ref_list'].apply(lambda x:('c' in x))
    constant_error = len(result_csv[(result_csv['constants'] ==True)&(result_csv['matching'] ==True)])
    constant_problem = len(result_csv[result_csv['constants'] ==True])
    print("{} : {}/{}={}".format("constant matching",constant_error
                          ,constant_problem,constant_error/constant_problem))
    # constant_error_MTDNN = len(result_csv[(result_csv['constants'] ==True)&(result_csv['bertmtdnn_ansids'] ==True)])
    # constant_problem_MTDNN = len(result_csv[result_csv['constants'] ==True])
    # print("{} : {}/{}={}".format("mtdnn_constant matching",constant_error_MTDNN
    #                       ,constant_problem_MTDNN,constant_error_MTDNN/constant_problem_MTDNN))

print("")