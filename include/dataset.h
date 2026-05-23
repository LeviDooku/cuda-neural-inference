//Pedro Velasco Santana
//File: dataset.h

//Purpose: generate synthetic data


#ifndef DATASET_H
#define DATASET_H

struct dataset{
    float* entries;
    float* tags;

    int num_samples;
    int num_entries;
};

void create_dataset(dataset& dataset, int num_samples);

void release_dataset(dataset& dataset);

void normalize_dataset(dataset& dataset);

void show_sample(const dataset& dataset, int index);

#endif
