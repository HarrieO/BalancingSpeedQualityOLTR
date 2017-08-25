# -*- coding: utf-8 -*-

import numpy as np
import os
import traceback


def cumulative(ranking, discount=0.9995):
    return np.cumsum(discount ** np.arange(ranking.shape[0]) * ranking)


def convert_time(time_in_seconds):
    seconds = time_in_seconds % 60
    minutes = time_in_seconds / 60 % 60
    hours = time_in_seconds / 3600
    return '%02d:%02d:%02d' % (hours, minutes, seconds)


def print_array(array):
    return ' '.join([str(x) for x in array] + ['\n'])


def create_folders(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


class OutputAverager(object):

    def __init__(self, simulation_arguments):
        self.average_folder = simulation_arguments.average_folder
        self._average_index = 0

    def create_average_file(self, simulation_output):
        output_file = simulation_output.output_path
        self.output_path = '%s/%s/%s.out' % (self.average_folder, simulation_output.dataset_name,
                                             simulation_output.simulation_name)
        click_model_names = []
        indices = {}
        have_indices = False
        has_nfeat = False
        click_model_test_runs = {}
        click_model_train_runs = {}
        click_model_times = {}
        click_model_nfeat_runs = {}
        click_model_other_runs = {}
        runs = []
        output_txt = ""
        try:
            with open(output_file, 'r') as f:
                print "opening %s" % output_file
                for line in f:
                    if "--------START--------" in line:
                        break
                    output_txt += line

                output_txt += "RESULTS\n"
                found = False

                for line in f:
                    if not found:
                        if line[:11] == "CLICK MODEL":
                            found = True
                            click_model = line.split()[2].split("_")[0]
                            cur_runs = {}
                            i_per_name = {}
                    else:
                        if line[:8] == "END TIME":     
                            found = False
                            time = int(line.split()[2])
                            if not have_indices and "N_FEAT" in indices:
                                has_nfeat = True
                            have_indices = True
                            if click_model not in click_model_times:
                                click_model_times[click_model] = []
                                click_model_test_runs[click_model] = []
                                click_model_train_runs[click_model] = []
                                click_model_nfeat_runs[click_model] = []
                            click_model_times[click_model].append(time)
                            for value_name in ["TRAIN", "TEST"]:
                                assert len(cur_runs[value_name]) == len(indices[value_name])
                            click_model_train_runs[click_model].append(cur_runs["TRAIN"])
                            click_model_test_runs[click_model].append(cur_runs["TEST"])
                            if "N_FEAT" in cur_runs:
                                assert len(cur_runs["N_FEAT"]) == len(indices["N_FEAT"])
                                click_model_nfeat_runs[click_model].append(cur_runs["N_FEAT"])
                            for name, values in cur_runs.items():
                                if not name in ['N_FEAT', 'TRAIN', 'TEST']:
                                    assert len(values) == len(indices[name])
                                    if not name in click_model_other_runs:
                                        click_model_other_runs[name] = {}
                                    if not click_model in click_model_other_runs[name]:
                                        click_model_other_runs[name][click_model] = []
                                    click_model_other_runs[name][click_model].append(values)
                        else:
                            splitted = line.split()
                            cur_i = int(splitted[0])
                            last_interleaving = 0
                            if splitted[-1] != 'None':
                                last_interleaving = int(splitted[-1])             
                            for i in range(len(splitted)/2-1):
                                name = splitted[1+i*2][:-1]
                                value = float(splitted[1+i*2+1])
                                i_per_name[name] = i_per_name.get(name,-1) + 1
                                if not have_indices:
                                    if name not in indices:
                                        indices[name] = []
                                    indices[name].append(cur_i)
                                else:
                                    assert indices[name][i_per_name[name]] == cur_i
                                if name not in cur_runs:
                                    cur_runs[name] = []
                                cur_runs[name].append(value)

                for click_model in click_model_times:
                    average_time = np.mean(click_model_times[click_model])
                    output_txt += "CLICK MODEL %s AVERAGE TIME %d SECONDS %s\n" \
                                  % (click_model, average_time, convert_time(average_time))


                output_txt += "TEST INDICES\n"
                output_txt += print_array(indices["TEST"])

                for click_model in click_model_times:
                    test_matrix = np.array(click_model_test_runs[click_model])
                    output_txt += "%s TEST MEAN\n" % click_model
                    output_txt += print_array(np.mean(test_matrix, axis=0))
                    output_txt += "%s TEST STD\n" % click_model
                    output_txt += print_array(np.std(test_matrix, axis=0))        

                output_txt += "TRAIN INDICES\n"
                output_txt += print_array(indices["TRAIN"])
                for click_model in click_model_times:
                    train_matrix = np.array(click_model_train_runs[click_model])
                    output_txt += "%s TRAIN MEAN\n" % click_model
                    output_txt += print_array(np.mean(train_matrix, axis=0))
                    output_txt += "%s TRAIN STD\n" % click_model
                    output_txt += print_array(np.std(train_matrix, axis=0))
                    for discount in [0.995, 0.9995]:
                        online_matrix = np.cumsum(train_matrix * discount**np.arange(train_matrix.shape[1]),axis=1)
                        output_txt += "%s TRAIN ONLINE MEAN %s\n" % (click_model,str(discount))
                        output_txt += print_array(np.mean(online_matrix, axis=0))
                        output_txt += "%s TRAIN ONLINE STD %s\n" % (click_model,str(discount))
                        output_txt += print_array(np.std(online_matrix, axis=0))

                if has_nfeat:
                    output_txt += "N_FEAT INDICES\n"
                    output_txt += print_array(indices["N_FEAT"])
                    for click_model in click_model_times:
                        nfeat_matrix = np.array(click_model_nfeat_runs[click_model])
                        output_txt += "%s N_FEAT MEAN\n" % click_model
                        output_txt += print_array(np.mean(nfeat_matrix, axis=0))
                        output_txt += "%s N_FEAT STD\n" % click_model
                        output_txt += print_array(np.std(nfeat_matrix, axis=0))

                for name, click_model_dict in click_model_other_runs.items():
                    output_txt += "%s INDICES\n" % name
                    output_txt += print_array(indices[name])
                    for click_model in click_model_times:
                        other_matrix = np.array(click_model_dict[click_model])
                        output_txt += "%s %s MEAN\n" % (click_model, name)
                        output_txt += print_array(np.mean(other_matrix, axis=0))
                        output_txt += "%s %s STD\n" % (click_model, name)
                        output_txt += print_array(np.std(other_matrix, axis=0))

                create_folders(self.output_path)
                with open(self.output_path, 'w') as w:
                    w.write(output_txt[:-1])
                    print 'Closed %d: %s on %s was averaged and stored.' % (self._average_index,
                            simulation_output.simulation_name, simulation_output.dataset_name)
        except KeyboardInterrupt:
            raise
        except Exception, err:
            traceback.print_exc()
            print 'Closed %d: FAILED processing %s on %s. (%s)' % (self._average_index,
                    simulation_output.simulation_name, simulation_output.dataset_name, output_file)
        self._average_index += 1
