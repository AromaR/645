
"""
Author: Jiang Li (jiangli@umass.edu)
"""

direct = [
    [558.2306578159332, 1829.773288011551, 3609.6431090831757, 6420.87073802948, 10211.48302102089, 14750.233685016632, 18895.740686178207, 22473.46243405342, 37383.61820411682, 66010.41671705246],
    [148.49930119514465, 556.3155546188354, 1374.5102760791779, 2505.2632491588593, 4316.55988574028, 8620.684856891632, 12723.731763839722, 16495.302155971527, 19386.604343891144, 26418.092468976974],
    [159.51824593544006, 879.5198590755463, 2953.755286693573, 6414.4703159332275, 11249.388313770294, 17896.54030108452, 23887.69083595276, 34033.491158008575, 39754.3114862442, 48458.457894802094],
    [209.26293301582336, 1088.0784902572632, 3484.9051311016083, 7141.136322021484, 11529.970471858978, 17434.795949220657, 22119.554977178574, 29521.03555703163, 35466.1697909832, 42928.570551872253]
]


# partition_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]

sampling_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sketch_refine = [
    [0.25913357734680176, 0.27359509468078613, 0.4111652374267578, 0.4184989929199219, 0.5668313503265381, 0.6619772911071777, 0.7182159423828125, 0.8892991542816162, 1.3644990921020508, 1.5913333892822266],
    [0.8669388294219971, 0.8386142253875732, 0.21471881866455078, 0.4449653625488281, 3.744262933731079, 0.6667013168334961, 2.6764731407165527, 0.7520754337310791, 6.625354528427124,6.010455131530762],
    [19.507932662963867, 40.663288831710815, 61.48417258262634, 87.54260039329529, 114.45325112342834, 142.56891512870789, 174.01130771636963, 208.06124305725098, 241.31188654899597, 278.71151089668274],
    [0.3793361186981201, 0.15805721282958984, 1.9514853954315186, 0.2092726230621338, 2.6171367168426514, 1.536609172821045, 0.3131873607635498, 6.021095275878906, 3.52630352973938, 0.9865024089813232]
]
'''trimmed out infeasible cases'''
sketch_refine = [
    [0.25913357734680176, 0.27359509468078613, 0.4111652374267578, 0.4184989929199219, 0.5668313503265381, 0.6619772911071777, 0.7182159423828125, 0.8892991542816162, 1.3644990921020508, 1.5913333892822266],
    [0.8669388294219971, 0.8386142253875732, None, None, 3.744262933731079, None, 2.6764731407165527, None, 6.625354528427124,6.010455131530762],
    [19.507932662963867, 40.663288831710815, 61.48417258262634, 87.54260039329529, 114.45325112342834, 142.56891512870789, 174.01130771636963, 208.06124305725098, 241.31188654899597, 278.71151089668274],
    [0.3793361186981201, None, 1.9514853954315186, None, 2.6171367168426514, 1.536609172821045, None, 6.021095275878906, 3.52630352973938, None]
]

import matplotlib.pyplot as plt

plt.figure()
for i in range(len(sketch_refine)):
    plt.subplot(1, len(sketch_refine), i + 1)
    if i == 0: plt.ylabel("Time (s)")
    sk = plt.plot(sampling_size, sketch_refine[i], color='black')
    plt.scatter(sampling_size, sketch_refine[i], color='black')
    dr = plt.plot(sampling_size, direct[i], '--', color='red')
    plt.title(f"Q{i + 1}")
    plt.yscale('log')
    plt.grid(axis="x")
    plt.xlabel("Sampled Proportion")

plt.legend(["Sketch Refine", "Direct"], loc='center', bbox_to_anchor=(1.3, 0.95))
plt.show()