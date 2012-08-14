#ifndef USER_COMPUTE_H_
#define USER_COMPUTE_H_

#include <fstream>

/*!\file Compute.h
 * \brief The computation procedures.
 */

/* getTempMeanVar()
 * Store mean and var in \p mean and \p var of temp \p data of size \p size.
 */
void getTempMeanVar(float* data, unsigned size, float& mean, float& var);

/* getContMeanVar()
 * Store the mean and var array in \p mean and \p var 
 * of cont \p data of size \p size.
 */
void getContMeanVar(float* data, unsigned size, unsigned window_size, 
                    float* mean, float* var);

/* clearDevData()
 * Clear the device data.
 */
void clearDevData(float* data, unsigned size);

/* calcCorr()
 * Calculate correlation.
 */
void calcCorr(float* corr, float* temp, float* cont,
              unsigned cont_size, unsigned temp_size,
              float temp_mean, float temp_var,
              float* cont_mean, float* cont_var);

/* stack()
 * Stack correlation
 */
void stack(float* corr, float* stack, 
           size_t corr_size, int stack_shift);

/* getMAD()
 * calculate the mad value 
 */
float* getMAD(float* data, unsigned size);

/* select()
 * Select points exceeds mad and output.
 */
void select(float* data, unsigned size, 
            float* mad, float ratio, 
            float sample_rate, unsigned num_valid_channel, 
            std::ofstream& out);

#endif /* USER_COMPUTE_H_ */
