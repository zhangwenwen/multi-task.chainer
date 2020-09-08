import matplotlib.pyplot as plt
import json
import argparse
import numpy as np
import time

def loadlog(path, keys=None):
    if keys is None or (isinstance(keys, list) and not keys) or not isinstance(keys, list):
        return []
    f = open(path)
    log_data = json.load(f)
    f.close()

    result = {}
    for key in keys:
        result[key] = []

    for log_data_item in log_data:
        for key in keys:
            if key in log_data_item.keys():
                result[key].append(log_data_item[key])

    return result


def exponent_average(result, alhpa=0.85):
    previous = None
    result_return = []
    for r in result:
        if previous is None:
            result_return.append(r)
            previous = r
        else:
            tmp = r * (1 - alhpa) + previous * alhpa
            result_return.append(tmp)
            previous = tmp
    return result_return


def repick(result):
    return_retult = []
    for r in result:
        if len(return_retult) == 0:
            return_retult.append(r)
        else:
            if not return_retult[-1] == r:
                return_retult.append(r)
    return return_retult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, default="log")
    args = parser.parse_args()

    # result = loadlog(args.log_path, ['main/loss/loc', 'main/loss/conf', 'main/loss/mask'])
    while True:

        result = loadlog("/home/andy/workspace/multi-task.chainer/result/final_voc/20190315_134220/log",
                         ['main/loss/loc', 'main/loss/conf', 'main/loss/mask', 'main/loss/split'])
        start =0
        plt.plot(result['main/loss/loc'][start:])
        plt.plot(exponent_average(result['main/loss/loc'][start:]))

        plt.plot(result['main/loss/conf'][start:])
        plt.plot(exponent_average(result['main/loss/conf'][start:]))

        plt.plot(result['main/loss/mask'][start:])
        plt.plot(exponent_average(result['main/loss/mask'][start:]))

        # plt.plot(result['main/loss/split'][start:])
        plt.plot(result['main/loss/split'][start:])
        #plt.plot(exponent_average(repick(result['main/loss/split'][start:])))
        #plt.plot(exponent_average(exponent_average(repick(result['main/loss/split'][start:]))))
        # plt.plot(np.array(exponent_average(repick(result['main/loss/split'][start:]))))
        # plt.plot(result1['main/loss/loc'])
        # plt.plot(result1['main/loss/conf'])
        # plt.plot(result1['main/loss/mask'])


        #plt.savefig('/home/andy/workspace/multi-task.chainer/log.png')
        plt.show()
        exit()
        time.sleep(600)



if __name__ == '__main__':
    main()
