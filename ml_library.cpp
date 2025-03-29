#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tbml {

// Forward declarations
class Matrix;
class DataSet;
class Model;
class Optimizer;

// ------------------- Utility Classes -------------------

// Matrix class for linear algebra operations
class Matrix {
public:
    Matrix() : m_rows(0), m_cols(0) {}
    
    Matrix(size_t rows, size_t cols, double initialValue = 0.0) 
        : m_rows(rows), m_cols(cols), m_data(rows * cols, initialValue) {}
    
    Matrix(const std::vector<std::vector<double>>& data) {
        if (data.empty()) {
            m_rows = 0;
            m_cols = 0;
            return;
        }
        
        m_rows = data.size();
        m_cols = data[0].size();
        m_data.resize(m_rows * m_cols);
        
        for (size_t i = 0; i < m_rows; ++i) {
            if (data[i].size() != m_cols) {
                throw std::invalid_argument("Inconsistent row sizes in matrix initialization");
            }
            for (size_t j = 0; j < m_cols; ++j) {
                at(i, j) = data[i][j];
            }
        }
    }
    
    double& at(size_t row, size_t col) {
        if (row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return m_data[row * m_cols + col];
    }
    
    const double& at(size_t row, size_t col) const {
        if (row >= m_rows || col >= m_cols) {
            throw std::out_of_range("Matrix indices out of range");
        }
        return m_data[row * m_cols + col];
    }
    
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    
    // Matrix-matrix addition
    Matrix operator+(const Matrix& other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(i, j) = at(i, j) + other.at(i, j);
            }
        }
        return result;
    }
    
    // Matrix-matrix subtraction
    Matrix operator-(const Matrix& other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(i, j) = at(i, j) - other.at(i, j);
            }
        }
        return result;
    }
    
    // Matrix-matrix multiplication
    Matrix operator*(const Matrix& other) const {
        if (m_cols != other.m_rows) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix result(m_rows, other.m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < other.m_cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < m_cols; ++k) {
                    sum += at(i, k) * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }
    
    // Scalar multiplication
    Matrix operator*(double scalar) const {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(i, j) = at(i, j) * scalar;
            }
        }
        return result;
    }
    
    // Element-wise multiplication (Hadamard product)
    Matrix hadamard(const Matrix& other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            throw std::invalid_argument("Matrix dimensions don't match for Hadamard product");
        }
        
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(i, j) = at(i, j) * other.at(i, j);
            }
        }
        return result;
    }
    
    // Transpose
    Matrix transpose() const {
        Matrix result(m_cols, m_rows);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(j, i) = at(i, j);
            }
        }
        return result;
    }
    
    // Initialize with random values
    void randomize(double min = -1.0, double max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);
        
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                at(i, j) = dist(gen);
            }
        }
    }
    
    // Apply a function to each element
    Matrix apply(const std::function<double(double)>& func) const {
        Matrix result(m_rows, m_cols);
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result.at(i, j) = func(at(i, j));
            }
        }
        return result;
    }
    
    // Get column as a vector
    std::vector<double> getColumn(size_t col) const {
        if (col >= m_cols) {
            throw std::out_of_range("Column index out of range");
        }
        
        std::vector<double> result(m_rows);
        for (size_t i = 0; i < m_rows; ++i) {
            result[i] = at(i, col);
        }
        return result;
    }
    
    // Get row as a vector
    std::vector<double> getRow(size_t row) const {
        if (row >= m_rows) {
            throw std::out_of_range("Row index out of range");
        }
        
        std::vector<double> result(m_cols);
        for (size_t j = 0; j < m_cols; ++j) {
            result[j] = at(row, j);
        }
        return result;
    }
    
    // Convert to vector of vectors (for interfacing with other code)
    std::vector<std::vector<double>> toVector() const {
        std::vector<std::vector<double>> result(m_rows, std::vector<double>(m_cols));
        for (size_t i = 0; i < m_rows; ++i) {
            for (size_t j = 0; j < m_cols; ++j) {
                result[i][j] = at(i, j);
            }
        }
        return result;
    }
    
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double> m_data;
};

// Dataset class for handling training and testing data
class DataSet {
public:
    DataSet() = default;
    
    // Constructor with features and labels
    DataSet(const Matrix& features, const Matrix& labels) 
        : m_features(features), m_labels(labels) {
        if (features.rows() != labels.rows()) {
            throw std::invalid_argument("Features and labels must have the same number of samples");
        }
    }
    
    // Load from vectors
    DataSet(const std::vector<std::vector<double>>& features, 
            const std::vector<std::vector<double>>& labels)
        : m_features(features), m_labels(labels) {
        if (m_features.rows() != m_labels.rows()) {
            throw std::invalid_argument("Features and labels must have the same number of samples");
        }
    }
    
    // Load from CSV file
    static DataSet fromCSV(const std::string& filename, bool hasHeader = true, 
                          char delimiter = ',', size_t labelCol = 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::vector<std::vector<double>> features;
        std::vector<std::vector<double>> labels;
        
        std::string line;
        if (hasHeader && std::getline(file, line)) {
            // Skip header
        }
        
        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, delimiter)) {
                try {
                    row.push_back(std::stod(cell));
                } catch (const std::exception& e) {
                    // Handle non-numeric value
                    row.push_back(0.0);
                }
            }
            
            if (!row.empty()) {
                std::vector<double> label = {row[labelCol]};
                row.erase(row.begin() + labelCol);
                
                features.push_back(row);
                labels.push_back(label);
            }
        }
        
        return DataSet(features, labels);
    }
    
    // Save to CSV file
    void toCSV(const std::string& filename, bool writeHeader = true, 
              char delimiter = ',', size_t labelCol = 0) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        if (writeHeader) {
            for (size_t j = 0; j < m_features.cols() + m_labels.cols(); ++j) {
                if (j > 0) {
                    file << delimiter;
                }
                if (j == labelCol) {
                    file << "Label";
                } else {
                    file << "Feature" << (j < labelCol ? j : j - 1);
                }
            }
            file << "\n";
        }
        
        for (size_t i = 0; i < m_features.rows(); ++i) {
            for (size_t j = 0; j < m_features.cols() + m_labels.cols(); ++j) {
                if (j > 0) {
                    file << delimiter;
                }
                
                if (j == labelCol) {
                    file << m_labels.at(i, 0);
                } else if (j < labelCol) {
                    file << m_features.at(i, j);
                } else {
                    file << m_features.at(i, j - 1);
                }
            }
            file << "\n";
        }
    }
    
    // Split data into training and testing sets
    std::pair<DataSet, DataSet> trainTestSplit(double testRatio = 0.2) const {
        if (testRatio < 0.0 || testRatio > 1.0) {
            throw std::invalid_argument("Test ratio must be between 0 and 1");
        }
        
        size_t numSamples = m_features.rows();
        size_t numTestSamples = static_cast<size_t>(numSamples * testRatio);
        size_t numTrainSamples = numSamples - numTestSamples;
        
        // Create indices and shuffle
        std::vector<size_t> indices(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            indices[i] = i;
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Create train and test datasets
        Matrix trainFeatures(numTrainSamples, m_features.cols());
        Matrix trainLabels(numTrainSamples, m_labels.cols());
        Matrix testFeatures(numTestSamples, m_features.cols());
        Matrix testLabels(numTestSamples, m_labels.cols());
        
        for (size_t i = 0; i < numTrainSamples; ++i) {
            size_t idx = indices[i];
            for (size_t j = 0; j < m_features.cols(); ++j) {
                trainFeatures.at(i, j) = m_features.at(idx, j);
            }
            for (size_t j = 0; j < m_labels.cols(); ++j) {
                trainLabels.at(i, j) = m_labels.at(idx, j);
            }
        }
        
        for (size_t i = 0; i < numTestSamples; ++i) {
            size_t idx = indices[numTrainSamples + i];
            for (size_t j = 0; j < m_features.cols(); ++j) {
                testFeatures.at(i, j) = m_features.at(idx, j);
            }
            for (size_t j = 0; j < m_labels.cols(); ++j) {
                testLabels.at(i, j) = m_labels.at(idx, j);
            }
        }
        
        return {DataSet(trainFeatures, trainLabels), DataSet(testFeatures, testLabels)};
    }
    
    // Get batch of data
    std::pair<Matrix, Matrix> getBatch(size_t batchSize, size_t batchIndex) const {
        size_t startIdx = batchIndex * batchSize;
        if (startIdx >= m_features.rows()) {
            throw std::out_of_range("Batch index out of range");
        }
        
        size_t actualBatchSize = std::min(batchSize, m_features.rows() - startIdx);
        
        Matrix batchFeatures(actualBatchSize, m_features.cols());
        Matrix batchLabels(actualBatchSize, m_labels.cols());
        
        for (size_t i = 0; i < actualBatchSize; ++i) {
            for (size_t j = 0; j < m_features.cols(); ++j) {
                batchFeatures.at(i, j) = m_features.at(startIdx + i, j);
            }
            for (size_t j = 0; j < m_labels.cols(); ++j) {
                batchLabels.at(i, j) = m_labels.at(startIdx + i, j);
            }
        }
        
        return {batchFeatures, batchLabels};
    }
    
    // Bootstrap sample (with replacement)
    DataSet bootstrapSample(size_t sampleSize = 0) const {
        if (sampleSize == 0) {
            sampleSize = m_features.rows();
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<size_t> dist(0, m_features.rows() - 1);
        
        Matrix sampleFeatures(sampleSize, m_features.cols());
        Matrix sampleLabels(sampleSize, m_labels.cols());
        
        for (size_t i = 0; i < sampleSize; ++i) {
            size_t idx = dist(g);
            
            for (size_t j = 0; j < m_features.cols(); ++j) {
                sampleFeatures.at(i, j) = m_features.at(idx, j);
            }
            
            for (size_t j = 0; j < m_labels.cols(); ++j) {
                sampleLabels.at(i, j) = m_labels.at(idx, j);
            }
        }
        
        return DataSet(sampleFeatures, sampleLabels);
    }
    
    size_t numSamples() const { return m_features.rows(); }
    size_t numFeatures() const { return m_features.cols(); }
    size_t numLabels() const { return m_labels.cols(); }
    
    const Matrix& features() const { return m_features; }
    const Matrix& labels() const { return m_labels; }
    
    void normalize() {
        for (size_t j = 0; j < m_features.cols(); ++j) {
            double sum = 0.0;
            double sumSquared = 0.0;
            
            // Calculate mean and variance
            for (size_t i = 0; i < m_features.rows(); ++i) {
                sum += m_features.at(i, j);
                sumSquared += m_features.at(i, j) * m_features.at(i, j);
            }
            
            double mean = sum / m_features.rows();
            double variance = (sumSquared / m_features.rows()) - (mean * mean);
            double stdDev = std::sqrt(variance);
            
            // Avoid division by zero
            stdDev = std::max(stdDev, 1e-10);
            
            // Normalize
            for (size_t i = 0; i < m_features.rows(); ++i) {
                m_features.at(i, j) = (m_features.at(i, j) - mean) / stdDev;
            }
        }
    }
    
private:
    Matrix m_features;
    Matrix m_labels;
};

// ------------------- Model Base Class -------------------

class Model {
public:
    virtual ~Model() = default;
    
    // Training method
    virtual void fit(const DataSet& dataset, size_t epochs = 100, size_t batchSize = 32) = 0;
    
    // Prediction method
    virtual Matrix predict(const Matrix& features) const = 0;
    
    // Evaluation metrics
    virtual double evaluate(const DataSet& dataset) const {
        // Default evaluation: mean squared error
        const Matrix& features = dataset.features();
        const Matrix& actualLabels = dataset.labels();
        
        Matrix predictedLabels = predict(features);
        
        double sumSquaredError = 0.0;
        for (size_t i = 0; i < actualLabels.rows(); ++i) {
            for (size_t j = 0; j < actualLabels.cols(); ++j) {
                double error = actualLabels.at(i, j) - predictedLabels.at(i, j);
                sumSquaredError += error * error;
            }
        }
        
        return sumSquaredError / (actualLabels.rows() * actualLabels.cols());
    }
    
    // Classification accuracy
    virtual double accuracy(const DataSet& dataset) const {
        const Matrix& features = dataset.features();
        const Matrix& actualLabels = dataset.labels();
        
        Matrix predictedLabels = predict(features);
        
        size_t correct = 0;
        for (size_t i = 0; i < actualLabels.rows(); ++i) {
            // For binary classification
            if (actualLabels.cols() == 1) {
                bool actual = actualLabels.at(i, 0) > 0.5;
                bool predicted = predictedLabels.at(i, 0) > 0.5;
                if (actual == predicted) {
                    ++correct;
                }
            } else {
                // For multiclass classification
                size_t actualMaxIdx = 0;
                size_t predictedMaxIdx = 0;
                
                for (size_t j = 1; j < actualLabels.cols(); ++j) {
                    if (actualLabels.at(i, j) > actualLabels.at(i, actualMaxIdx)) {
                        actualMaxIdx = j;
                    }
                    if (predictedLabels.at(i, j) > predictedLabels.at(i, predictedMaxIdx)) {
                        predictedMaxIdx = j;
                    }
                }
                
                if (actualMaxIdx == predictedMaxIdx) {
                    ++correct;
                }
            }
        }
        
        return static_cast<double>(correct) / actualLabels.rows();
    }
    
    // Save and load model parameters
    virtual void saveModel(const std::string& filename) const = 0;
    virtual void loadModel(const std::string& filename) = 0;
};

// ------------------- Optimizer Classes -------------------

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual Matrix updateParameters(const Matrix& parameters, const Matrix& gradients, size_t iteration) = 0;
};

class SGDOptimizer : public Optimizer {
public:
    explicit SGDOptimizer(double learningRate = 0.01, double momentum = 0.0)
        : m_learningRate(learningRate), m_momentum(momentum), m_velocity(0, 0) {}
    
    Matrix updateParameters(const Matrix& parameters, const Matrix& gradients, size_t) override {
        if (m_velocity.rows() == 0) {
            // Initialize velocity matrix
            m_velocity = Matrix(parameters.rows(), parameters.cols(), 0.0);
        }
        
        // Update velocity with momentum
        m_velocity = (m_velocity * m_momentum) + (gradients * (-m_learningRate));
        
        // Return updated parameters
        return parameters + m_velocity;
    }
    
private:
    double m_learningRate;
    double m_momentum;
    Matrix m_velocity;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), 
          m_m(0, 0), m_v(0, 0) {}
    
    Matrix updateParameters(const Matrix& parameters, const Matrix& gradients, size_t iteration) override {
        if (m_m.rows() == 0) {
            // Initialize first and second moment vectors
            m_m = Matrix(parameters.rows(), parameters.cols(), 0.0);
            m_v = Matrix(parameters.rows(), parameters.cols(), 0.0);
        }
        
        iteration += 1; // Avoid division by zero in bias correction
        
        // Update biased first moment estimate
        m_m = m_m * m_beta1 + gradients * (1.0 - m_beta1);
        
        // Update biased second raw moment estimate
        Matrix gradSquared = gradients.apply([](double x) { return x * x; });
        m_v = m_v * m_beta2 + gradSquared * (1.0 - m_beta2);
        
        // Compute bias-corrected first moment estimate
        Matrix mCorrected = m_m * (1.0 / (1.0 - std::pow(m_beta1, iteration)));
        
        // Compute bias-corrected second raw moment estimate
        Matrix vCorrected = m_v * (1.0 / (1.0 - std::pow(m_beta2, iteration)));
        
        // Prepare update
        Matrix update(parameters.rows(), parameters.cols());
        for (size_t i = 0; i < parameters.rows(); ++i) {
            for (size_t j = 0; j < parameters.cols(); ++j) {
                update.at(i, j) = -m_learningRate * mCorrected.at(i, j) / 
                                 (std::sqrt(vCorrected.at(i, j)) + m_epsilon);
            }
        }
        
        return parameters + update;
    }
    
private:
    double m_learningRate;
    double m_beta1;
    double m_beta2;
    double m_epsilon;
    Matrix m_m; // First moment estimate
    Matrix m_v; // Second moment estimate
};

// ------------------- Activation Functions -------------------

namespace activation {

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Softmax function (applied to a row of values)
std::vector<double> softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    
    // Find maximum value for numerical stability
    double maxVal = *std::max_element(x.begin(), x.end());
    
    // Compute exp for each value and sum
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - maxVal);
        sum += result[i];
    }
    
    // Normalize
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }
    
    return result;
}

// Apply softmax to each row of a matrix
Matrix softmaxMatrix(const Matrix& matrix) {
    Matrix result(matrix.rows(), matrix.cols());
    
    for (size_t i = 0; i < matrix.rows(); ++i) {
        std::vector<double> row(matrix.cols());
        for (size_t j = 0; j < matrix.cols(); ++j) {
            row[j] = matrix.at(i, j);
        }
        
        std::vector<double> softmaxRow = softmax(row);
        
        for (size_t j = 0; j < matrix.cols(); ++j) {
            result.at(i, j) = softmaxRow[j];
        }
    }
    
    return result;
}

} // namespace activation

// ------------------- Loss Functions -------------------

namespace loss {

// Mean Squared Error loss
double mse(const Matrix& predicted, const Matrix& actual) {
    if (predicted.rows() != actual.rows() || predicted.cols() != actual.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for MSE calculation");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < predicted.rows(); ++i) {
        for (size_t j = 0; j < predicted.cols(); ++j) {
            double diff = predicted.at(i, j) - actual.at(i, j);
            sum += diff * diff;
        }
    }
    
    return sum / (predicted.rows() * predicted.cols());
}

// Binary Cross-Entropy loss
double binaryCrossEntropy(const Matrix& predicted, const Matrix& actual) {
    if (predicted.rows() != actual.rows() || predicted.cols() != actual.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for BCE calculation");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < predicted.rows(); ++i) {
        for (size_t j = 0; j < predicted.cols(); ++j) {
            double p = std::max(std::min(predicted.at(i, j), 1.0 - 1e-15), 1e-15); // Clip to avoid log(0)
            double a = actual.at(i, j);
            sum += a * std::log(p) + (1.0 - a) * std::log(1.0 - p);
        }
    }
    
    return -sum / predicted.rows();
}

// Categorical Cross-Entropy loss
double categoricalCrossEntropy(const Matrix& predicted, const Matrix& actual) {
    if (predicted.rows() != actual.rows() || predicted.cols() != actual.cols()) {
        throw std::invalid_argument("Matrix dimensions don't match for CCE calculation");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < predicted.rows(); ++i) {
        for (size_t j = 0; j < predicted.cols(); ++j) {
            double p = std::max(predicted.at(i, j), 1e-15); // Clip to avoid log(0)
            double a = actual.at(i, j);
            if (a > 0) { // Only calculate for positive labels (one-hot encoded)
                sum += a * std::log(p);
            }
        }
    }
    
    return -sum / predicted.rows();
}

} // namespace loss

// ------------------- Model Implementations -------------------

// Linear Regression Implementation
class LinearRegression : public Model {
public:
    LinearRegression(size_t inputDim, std::shared_ptr<Optimizer> optimizer = nullptr)
        : m_weights(1, inputDim), m_bias(0.0),
          m_optimizer(optimizer ? optimizer : std::make_shared<SGDOptimizer>()) {
        // Initialize weights
        m_weights.randomize(-0.1, 0.1);
    }
    
    void fit(const DataSet& dataset, size_t epochs = 100, size_t batchSize = 32) override {
        const Matrix& features = dataset.features();
        const Matrix& labels = dataset.labels();
        
        if (features.cols() != m_weights.cols()) {
            throw std::invalid_argument("Feature dimension mismatch");
        }
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            size_t numBatches = (dataset.numSamples() + batchSize - 1) / batchSize;
            
            double epochLoss = 0.0;
            for (size_t batch = 0; batch < numBatches; ++batch) {
                auto [batchFeatures, batchLabels] = dataset.getBatch(batchSize, batch);
                
                // Forward pass
                Matrix predictions = predict(batchFeatures);
                
                // Compute loss
                double batchLoss = loss::mse(predictions, batchLabels);
                epochLoss += batchLoss;
                
                // Compute gradients
                Matrix error = predictions - batchLabels;
                Matrix weightGradients(1, m_weights.cols(), 0.0);
                double biasGradient = 0.0;
                
                for (size_t i = 0; i < error.rows(); ++i) {
                    for (size_t j = 0; j < m_weights.cols(); ++j) {
                        weightGradients.at(0, j) += error.at(i, 0) * batchFeatures.at(i, j);
                    }
                    biasGradient += error.at(i, 0);
                }
                
                // Normalize gradients by batch size
                weightGradients = weightGradients * (1.0 / batchFeatures.rows());
                biasGradient /= batchFeatures.rows();
                
                // Update weights and bias
                m_weights = m_optimizer->updateParameters(m_weights, weightGradients, epoch * numBatches + batch);
                m_bias -= 0.01 * biasGradient; // Simple update for bias
            }
            
            // Log progress
            if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << ", Loss: " << (epochLoss / numBatches) << std::endl;
            }
        }
    }
    
    Matrix predict(const Matrix& features) const override {
        Matrix result(features.rows(), 1);
        
        for (size_t i = 0; i < features.rows(); ++i) {
            double sum = m_bias;
            for (size_t j = 0; j < features.cols(); ++j) {
                sum += features.at(i, j) * m_weights.at(0, j);
            }
            result.at(i, 0) = sum;
        }
        
        return result;
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save bias
        file << m_bias << "\n";
        
        // Save weights
        for (size_t j = 0; j < m_weights.cols(); ++j) {
            file << m_weights.at(0, j);
            if (j < m_weights.cols() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Load bias
        std::string line;
        if (std::getline(file, line)) {
            m_bias = std::stod(line);
        }
        
        // Load weights
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> weights;
            
            while (std::getline(ss, cell, ',')) {
                weights.push_back(std::stod(cell));
            }
            
            if (weights.size() != m_weights.cols()) {
                throw std::runtime_error("Weight dimension mismatch in loaded model");
            }
            
            for (size_t j = 0; j < weights.size(); ++j) {
                m_weights.at(0, j) = weights[j];
            }
        }
    }
    
    // Getter for weights
    const Matrix& getWeights() const {
        return m_weights;
    }
    
    // Getter for bias
    double getBias() const {
        return m_bias;
    }
    
private:
    Matrix m_weights;
    double m_bias;
    std::shared_ptr<Optimizer> m_optimizer;
};

// Logistic Regression Implementation
class LogisticRegression : public Model {
public:
    LogisticRegression(size_t inputDim, size_t numClasses = 1, 
                     std::shared_ptr<Optimizer> optimizer = nullptr)
        : m_weights(numClasses, inputDim), m_bias(1, numClasses),
          m_optimizer(optimizer ? optimizer : std::make_shared<SGDOptimizer>(0.01)),
          m_numClasses(numClasses) {
        // Initialize weights and bias
        m_weights.randomize(-0.1, 0.1);
        m_bias.randomize(-0.1, 0.1);
    }
    
    void fit(const DataSet& dataset, size_t epochs = 100, size_t batchSize = 32) override {
        const Matrix& features = dataset.features();
        const Matrix& labels = dataset.labels();
        
        if (features.cols() != m_weights.cols()) {
            throw std::invalid_argument("Feature dimension mismatch");
        }
        
        if (labels.cols() != m_numClasses) {
            throw std::invalid_argument("Label dimension mismatch");
        }
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            size_t numBatches = (dataset.numSamples() + batchSize - 1) / batchSize;
            
            double epochLoss = 0.0;
            for (size_t batch = 0; batch < numBatches; ++batch) {
                auto [batchFeatures, batchLabels] = dataset.getBatch(batchSize, batch);
                
                // Forward pass
                Matrix predictions = predict(batchFeatures);
                
                // Compute loss
                double batchLoss;
                if (m_numClasses == 1) {
                    batchLoss = loss::binaryCrossEntropy(predictions, batchLabels);
                } else {
                    batchLoss = loss::categoricalCrossEntropy(predictions, batchLabels);
                }
                epochLoss += batchLoss;
                
                // Compute gradients
                Matrix error = predictions - batchLabels;
                Matrix weightGradients = Matrix(m_weights.rows(), m_weights.cols(), 0.0);
                Matrix biasGradients = Matrix(1, m_numClasses, 0.0);
                
                for (size_t i = 0; i < error.rows(); ++i) {
                    for (size_t c = 0; c < m_numClasses; ++c) {
                        for (size_t j = 0; j < m_weights.cols(); ++j) {
                            weightGradients.at(c, j) += error.at(i, c) * batchFeatures.at(i, j);
                        }
                        biasGradients.at(0, c) += error.at(i, c);
                    }
                }
                
                // Normalize gradients by batch size
                weightGradients = weightGradients * (1.0 / batchFeatures.rows());
                biasGradients = biasGradients * (1.0 / batchFeatures.rows());
                
                // Update weights and bias
                m_weights = m_optimizer->updateParameters(m_weights, weightGradients, epoch * numBatches + batch);
                m_bias = m_optimizer->updateParameters(m_bias, biasGradients, epoch * numBatches + batch);
            }
            
            // Log progress
            if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1) {
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                          << ", Loss: " << (epochLoss / numBatches) << std::endl;
            }
        }
    }
    
    Matrix predict(const Matrix& features) const override {
        // Linear combination: X * W^T + b
        Matrix linearOutput(features.rows(), m_numClasses);
        
        for (size_t i = 0; i < features.rows(); ++i) {
            for (size_t c = 0; c < m_numClasses; ++c) {
                double sum = m_bias.at(0, c);
                for (size_t j = 0; j < features.cols(); ++j) {
                    sum += features.at(i, j) * m_weights.at(c, j);
                }
                linearOutput.at(i, c) = sum;
            }
        }
        
        // Apply sigmoid for binary classification or softmax for multiclass
        if (m_numClasses == 1) {
            // Binary classification
            return linearOutput.apply(activation::sigmoid);
        } else {
            // Multiclass classification
            return activation::softmaxMatrix(linearOutput);
        }
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save num classes
        file << m_numClasses << "\n";
        
        // Save bias
        for (size_t c = 0; c < m_numClasses; ++c) {
            file << m_bias.at(0, c);
            if (c < m_numClasses - 1) {
                file << ",";
            }
        }
        file << "\n";
        
        // Save weights
        for (size_t c = 0; c < m_numClasses; ++c) {
            for (size_t j = 0; j < m_weights.cols(); ++j) {
                file << m_weights.at(c, j);
                if (j < m_weights.cols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Load num classes
        std::string line;
        if (std::getline(file, line)) {
            size_t numClasses = std::stoul(line);
            if (numClasses != m_numClasses) {
                throw std::runtime_error("Class count mismatch in loaded model");
            }
        }
        
        // Load bias
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            size_t c = 0;
            
            while (std::getline(ss, cell, ',') && c < m_numClasses) {
                m_bias.at(0, c++) = std::stod(cell);
            }
        }
        
        // Load weights
        for (size_t c = 0; c < m_numClasses; ++c) {
            if (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                size_t j = 0;
                
                while (std::getline(ss, cell, ',') && j < m_weights.cols()) {
                    m_weights.at(c, j++) = std::stod(cell);
                }
                
                if (j != m_weights.cols()) {
                    throw std::runtime_error("Weight dimension mismatch in loaded model");
                }
            }
        }
    }
    
    // Getter for weights
    const Matrix& getWeights() const {
        return m_weights;
    }
    
    // Getter for bias
    const Matrix& getBias() const {
        return m_bias;
    }
    
private:
    Matrix m_weights;
    Matrix m_bias;
    std::shared_ptr<Optimizer> m_optimizer;
    size_t m_numClasses;
};

// K-means Clustering Implementation
class KMeansClustering : public Model {
public:
    explicit KMeansClustering(size_t numClusters, size_t maxIterations = 100)
        : m_numClusters(numClusters), m_maxIterations(maxIterations), m_centroids(numClusters, 0) {}
    
    void fit(const DataSet& dataset, size_t epochs = 1, size_t = 32) override {
        const Matrix& features = dataset.features();
        if (features.rows() < m_numClusters) {
            throw std::invalid_argument("Number of samples must be greater than number of clusters");
        }
        
        // Initialize centroids with random samples
        std::vector<size_t> indices(features.rows());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        m_centroids = Matrix(m_numClusters, features.cols());
        for (size_t i = 0; i < m_numClusters; ++i) {
            for (size_t j = 0; j < features.cols(); ++j) {
                m_centroids.at(i, j) = features.at(indices[i], j);
            }
        }
        
        // K-means algorithm
        std::vector<size_t> clusterAssignments(features.rows());
        bool converged = false;
        size_t iteration = 0;
        
        while (!converged && iteration < m_maxIterations * epochs) {
            // Assign each point to the nearest centroid
            bool assignmentsChanged = false;
            for (size_t i = 0; i < features.rows(); ++i) {
                size_t closestCluster = 0;
                double minDistance = std::numeric_limits<double>::max();
                
                for (size_t k = 0; k < m_numClusters; ++k) {
                    double distance = 0.0;
                    for (size_t j = 0; j < features.cols(); ++j) {
                        double diff = features.at(i, j) - m_centroids.at(k, j);
                        distance += diff * diff;
                    }
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestCluster = k;
                    }
                }
                
                if (clusterAssignments[i] != closestCluster) {
                    assignmentsChanged = true;
                    clusterAssignments[i] = closestCluster;
                }
            }
            
            // Update centroids
            Matrix newCentroids(m_numClusters, features.cols(), 0.0);
            std::vector<size_t> clusterSizes(m_numClusters, 0);
            
            for (size_t i = 0; i < features.rows(); ++i) {
                size_t cluster = clusterAssignments[i];
                clusterSizes[cluster]++;
                
                for (size_t j = 0; j < features.cols(); ++j) {
                    newCentroids.at(cluster, j) += features.at(i, j);
                }
            }
            
            for (size_t k = 0; k < m_numClusters; ++k) {
                if (clusterSizes[k] > 0) {
                    for (size_t j = 0; j < features.cols(); ++j) {
                        newCentroids.at(k, j) /= clusterSizes[k];
                    }
                }
            }
            
            // Check convergence
            converged = !assignmentsChanged;
            m_centroids = newCentroids;
            
            iteration++;
            if (iteration % 10 == 0 || iteration == 1 || converged) {
                std::cout << "Iteration " << iteration << ", Converged: " << (converged ? "Yes" : "No") << std::endl;
            }
        }
        
        // Store cluster assignments
        m_clusterAssignments = clusterAssignments;
    }
    
    Matrix predict(const Matrix& features) const override {
        Matrix result(features.rows(), 1);
        
        for (size_t i = 0; i < features.rows(); ++i) {
            size_t closestCluster = 0;
            double minDistance = std::numeric_limits<double>::max();
            
            for (size_t k = 0; k < m_numClusters; ++k) {
                double distance = 0.0;
                for (size_t j = 0; j < features.cols(); ++j) {
                    double diff = features.at(i, j) - m_centroids.at(k, j);
                    distance += diff * diff;
                }
                
                if (distance < minDistance) {
                    minDistance = distance;
                    closestCluster = k;
                }
            }
            
            result.at(i, 0) = static_cast<double>(closestCluster);
        }
        
        return result;
    }
    
    double evaluate(const DataSet& dataset) const override {
        // Use inertia (sum of squared distances to centroids) as evaluation metric
        const Matrix& features = dataset.features();
        Matrix clusterAssignments = predict(features);
        
        double inertia = 0.0;
        for (size_t i = 0; i < features.rows(); ++i) {
            size_t cluster = static_cast<size_t>(clusterAssignments.at(i, 0));
            
            for (size_t j = 0; j < features.cols(); ++j) {
                double diff = features.at(i, j) - m_centroids.at(cluster, j);
                inertia += diff * diff;
            }
        }
        
        return inertia;
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save number of clusters and feature dimension
        file << m_numClusters << "\n";
        file << m_centroids.cols() << "\n";
        
        // Save centroids
        for (size_t k = 0; k < m_numClusters; ++k) {
            for (size_t j = 0; j < m_centroids.cols(); ++j) {
                file << m_centroids.at(k, j);
                if (j < m_centroids.cols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        std::string line;
        
        // Load number of clusters
        if (std::getline(file, line)) {
            size_t numClusters = std::stoul(line);
            if (numClusters != m_numClusters) {
                m_numClusters = numClusters;
            }
        }
        
        // Load feature dimension
        if (std::getline(file, line)) {
            size_t featureDim = std::stoul(line);
            m_centroids = Matrix(m_numClusters, featureDim);
        }
        
        // Load centroids
        for (size_t k = 0; k < m_numClusters; ++k) {
            if (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                size_t j = 0;
                
                while (std::getline(ss, cell, ',') && j < m_centroids.cols()) {
                    m_centroids.at(k, j++) = std::stod(cell);
                }
                
                if (j != m_centroids.cols()) {
                    throw std::runtime_error("Centroid dimension mismatch in loaded model");
                }
            }
        }
    }
    
    const Matrix& getCentroids() const {
        return m_centroids;
    }
    
    const std::vector<size_t>& getClusterAssignments() const {
        return m_clusterAssignments;
    }
    
private:
    size_t m_numClusters;
    size_t m_maxIterations;
    Matrix m_centroids;
    std::vector<size_t> m_clusterAssignments;
};

// ------------------- Decision Tree and Random Forest -------------------

// Decision Tree Node
struct DecisionTreeNode {
    bool isLeaf = false;
    size_t featureIndex = 0;
    double threshold = 0.0;
    double prediction = 0.0;
    std::unique_ptr<DecisionTreeNode> leftChild;
    std::unique_ptr<DecisionTreeNode> rightChild;
    
    // For tree serialization
    void serialize(std::ostream& os) const {
        os << (isLeaf ? 1 : 0) << ",";
        os << featureIndex << ",";
        os << threshold << ",";
        os << prediction << "\n";
        
        if (!isLeaf) {
            if (leftChild) {
                leftChild->serialize(os);
            } else {
                os << "null\n";
            }
            
            if (rightChild) {
                rightChild->serialize(os);
            } else {
                os << "null\n";
            }
        }
    }
    
    static std::unique_ptr<DecisionTreeNode> deserialize(std::istream& is) {
        std::string line;
        if (!std::getline(is, line) || line == "null") {
            return nullptr;
        }
        
        std::stringstream ss(line);
        std::string cell;
        auto node = std::make_unique<DecisionTreeNode>();
        
        if (std::getline(ss, cell, ',')) {
            node->isLeaf = (std::stoi(cell) == 1);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->featureIndex = std::stoul(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->threshold = std::stod(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->prediction = std::stod(cell);
        }
        
        if (!node->isLeaf) {
            node->leftChild = deserialize(is);
            node->rightChild = deserialize(is);
        }
        
        return node;
    }
};

// Decision Tree Regressor
class DecisionTreeRegressor {
public:
    DecisionTreeRegressor(size_t maxDepth = 10, size_t minSamplesSplit = 2, double minImpurityDecrease = 0.0)
        : m_maxDepth(maxDepth), m_minSamplesSplit(minSamplesSplit), 
          m_minImpurityDecrease(minImpurityDecrease), m_root(nullptr) {}
    
    void fit(const Matrix& features, const Matrix& targets) {
        // Create root node
        m_root = buildTree(features, targets, 0);
    }
    
    Matrix predict(const Matrix& features) const {
        Matrix predictions(features.rows(), 1);
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions.at(i, 0) = predictSample(features, i);
        }
        return predictions;
    }
    
    // Serialize tree to a file
    void save(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_maxDepth << "," << m_minSamplesSplit << "," << m_minImpurityDecrease << "\n";
        
        // Save tree structure
        if (m_root) {
            m_root->serialize(file);
        } else {
            file << "null\n";
        }
    }
    
    // Deserialize tree from a file
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minSamplesSplit = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minImpurityDecrease = std::stod(cell);
            }
        }
        
        // Load tree structure
        m_root = DecisionTreeNode::deserialize(file);
    }
    
private:
    size_t m_maxDepth;
    size_t m_minSamplesSplit;
    double m_minImpurityDecrease;
    std::unique_ptr<DecisionTreeNode> m_root;
    
    double predictSample(const Matrix& features, size_t sampleIndex) const {
        const DecisionTreeNode* node = m_root.get();
        while (node && !node->isLeaf) {
            if (features.at(sampleIndex, node->featureIndex) <= node->threshold) {
                node = node->leftChild.get();
            } else {
                node = node->rightChild.get();
            }
        }
        return node ? node->prediction : 0.0;
    }
    
    // Calculate mean value of targets
    double calculateMean(const Matrix& targets) const {
        double sum = 0.0;
        for (size_t i = 0; i < targets.rows(); ++i) {
            sum += targets.at(i, 0);
        }
        return sum / targets.rows();
    }
    
    // Calculate mean squared error
    double calculateMSE(const Matrix& targets, double prediction) const {
        double mse = 0.0;
        for (size_t i = 0; i < targets.rows(); ++i) {
            double diff = targets.at(i, 0) - prediction;
            mse += diff * diff;
        }
        return mse / targets.rows();
    }
    
    // Find best split for a node
    std::tuple<size_t, double, double, double> findBestSplit(
        const Matrix& features, const Matrix& targets) const {
        
        size_t bestFeatureIndex = 0;
        double bestThreshold = 0.0;
        double bestScore = std::numeric_limits<double>::max();
        double bestImpurityDecrease = 0.0;
        
        double currentImpurity = calculateMSE(targets, calculateMean(targets));
        
        for (size_t featureIndex = 0; featureIndex < features.cols(); ++featureIndex) {
            // Get unique values for the feature
            std::set<double> uniqueValues;
            for (size_t i = 0; i < features.rows(); ++i) {
                uniqueValues.insert(features.at(i, featureIndex));
            }
            
            // Try each unique value as threshold
            for (double threshold : uniqueValues) {
                // Split samples
                std::vector<size_t> leftIndices, rightIndices;
                for (size_t i = 0; i < features.rows(); ++i) {
                    if (features.at(i, featureIndex) <= threshold) {
                        leftIndices.push_back(i);
                    } else {
                        rightIndices.push_back(i);
                    }
                }
                
                // Skip if split doesn't meet minimum samples criteria
                if (leftIndices.size() < m_minSamplesSplit || rightIndices.size() < m_minSamplesSplit) {
                    continue;
                }
                
                // Create left and right target matrices
                Matrix leftTargets(leftIndices.size(), 1);
                Matrix rightTargets(rightIndices.size(), 1);
                
                for (size_t i = 0; i < leftIndices.size(); ++i) {
                    leftTargets.at(i, 0) = targets.at(leftIndices[i], 0);
                }
                
                for (size_t i = 0; i < rightIndices.size(); ++i) {
                    rightTargets.at(i, 0) = targets.at(rightIndices[i], 0);
                }
                
                // Calculate means and impurities
                double leftMean = calculateMean(leftTargets);
                double rightMean = calculateMean(rightTargets);
                double leftImpurity = calculateMSE(leftTargets, leftMean);
                double rightImpurity = calculateMSE(rightTargets, rightMean);
                
                // Weighted impurity
                double leftWeight = static_cast<double>(leftIndices.size()) / features.rows();
                double rightWeight = static_cast<double>(rightIndices.size()) / features.rows();
                double weightedImpurity = leftWeight * leftImpurity + rightWeight * rightImpurity;
                
                // Calculate impurity decrease
                double impurityDecrease = currentImpurity - weightedImpurity;
                
                // Update best if this split is better
                if (impurityDecrease > m_minImpurityDecrease && weightedImpurity < bestScore) {
                    bestScore = weightedImpurity;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                    bestImpurityDecrease = impurityDecrease;
                }
            }
        }
        
        return {bestFeatureIndex, bestThreshold, bestScore, bestImpurityDecrease};
    }
    
    // Recursively build tree
    std::unique_ptr<DecisionTreeNode> buildTree(
        const Matrix& features, const Matrix& targets, size_t depth) {
        
        auto node = std::make_unique<DecisionTreeNode>();
        
        // Stop criteria
        if (depth >= m_maxDepth || features.rows() <= m_minSamplesSplit) {
            node->isLeaf = true;
            node->prediction = calculateMean(targets);
            return node;
        }
        
        // Find best split
        auto [featureIndex, threshold, score, impurityDecrease] = findBestSplit(features, targets);
        
        // If no good split is found, make a leaf node
        if (impurityDecrease <= m_minImpurityDecrease) {
            node->isLeaf = true;
            node->prediction = calculateMean(targets);
            return node;
        }
        
        // Set node parameters
        node->isLeaf = false;
        node->featureIndex = featureIndex;
        node->threshold = threshold;
        
        // Split samples
        std::vector<size_t> leftIndices, rightIndices;
        for (size_t i = 0; i < features.rows(); ++i) {
            if (features.at(i, featureIndex) <= threshold) {
                leftIndices.push_back(i);
            } else {
                rightIndices.push_back(i);
            }
        }
        
        // Create left and right feature and target matrices
        Matrix leftFeatures(leftIndices.size(), features.cols());
        Matrix leftTargets(leftIndices.size(), 1);
        Matrix rightFeatures(rightIndices.size(), features.cols());
        Matrix rightTargets(rightIndices.size(), 1);
        
        for (size_t i = 0; i < leftIndices.size(); ++i) {
            leftTargets.at(i, 0) = targets.at(leftIndices[i], 0);
            for (size_t j = 0; j < features.cols(); ++j) {
                leftFeatures.at(i, j) = features.at(leftIndices[i], j);
            }
        }
        
        for (size_t i = 0; i < rightIndices.size(); ++i) {
            rightTargets.at(i, 0) = targets.at(rightIndices[i], 0);
            for (size_t j = 0; j < features.cols(); ++j) {
                rightFeatures.at(i, j) = features.at(rightIndices[i], j);
            }
        }
        
        // Recursively build subtrees
        node->leftChild = buildTree(leftFeatures, leftTargets, depth + 1);
        node->rightChild = buildTree(rightFeatures, rightTargets, depth + 1);
        
        return node;
    }
};

// Random Forest Regressor
class RandomForestRegressor : public Model {
public:
    RandomForestRegressor(size_t numTrees = 100, size_t maxDepth = 10, 
                         size_t minSamplesSplit = 2, double maxFeatures = 0.3)
        : m_numTrees(numTrees), m_maxDepth(maxDepth), 
          m_minSamplesSplit(minSamplesSplit), m_maxFeatures(maxFeatures) {}
    
    void fit(const DataSet& dataset, size_t epochs = 1, size_t = 32) override {
        // Clear existing trees
        m_trees.clear();
        m_treeFeatures.clear();
        
        const Matrix& features = dataset.features();
        const Matrix& targets = dataset.labels();
        
        if (targets.cols() != 1) {
            throw std::invalid_argument("Random Forest currently only supports single target regression");
        }
        
        // Calculate actual number of features to use
        size_t numFeaturesToUse = static_cast<size_t>(features.cols() * m_maxFeatures);
        numFeaturesToUse = std::max(numFeaturesToUse, size_t(1));
        
        for (size_t i = 0; i < m_numTrees * epochs; ++i) {
            // Bootstrap sample
            std::vector<size_t> sampleIndices = bootstrapSample(features.rows());
            
            // Select features for this tree
            std::vector<size_t> featureIndices = selectFeatures(features.cols(), numFeaturesToUse);
            
            // Create feature and target matrices for this tree
            Matrix treeFeatures(sampleIndices.size(), featureIndices.size());
            Matrix treeTargets(sampleIndices.size(), 1);
            
            for (size_t j = 0; j < sampleIndices.size(); ++j) {
                treeTargets.at(j, 0) = targets.at(sampleIndices[j], 0);
                for (size_t k = 0; k < featureIndices.size(); ++k) {
                    treeFeatures.at(j, k) = features.at(sampleIndices[j], featureIndices[k]);
                }
            }
            
            // Train tree
            auto tree = std::make_unique<DecisionTreeRegressor>(m_maxDepth, m_minSamplesSplit);
            tree->fit(treeFeatures, treeTargets);
            
            // Store tree and its feature indices
            m_trees.push_back(std::move(tree));
            m_treeFeatures.push_back(featureIndices);
            
            if ((i + 1) % 10 == 0 || i == 0 || i == m_numTrees * epochs - 1) {
                std::cout << "Trained tree " << (i + 1) << "/" << (m_numTrees * epochs) << std::endl;
            }
        }
    }
    
    Matrix predict(const Matrix& features) const override {
        Matrix predictions(features.rows(), 1, 0.0);
        
        // Get predictions from each tree
        for (size_t i = 0; i < m_trees.size(); ++i) {
            // Select features for this tree
            Matrix treeFeatures(features.rows(), m_treeFeatures[i].size());
            for (size_t j = 0; j < features.rows(); ++j) {
                for (size_t k = 0; k < m_treeFeatures[i].size(); ++k) {
                    treeFeatures.at(j, k) = features.at(j, m_treeFeatures[i][k]);
                }
            }
            
            // Get tree predictions
            Matrix treePredictions = m_trees[i]->predict(treeFeatures);
            
            // Accumulate predictions
            for (size_t j = 0; j < features.rows(); ++j) {
                predictions.at(j, 0) += treePredictions.at(j, 0);
            }
        }
        
        // Average predictions
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions.at(i, 0) /= m_trees.size();
        }
        
        return predictions;
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_numTrees << "," << m_maxDepth << "," << m_minSamplesSplit << "," << m_maxFeatures << "\n";
        
        // Save number of trees and feature information
        file << m_trees.size() << "\n";
        
        // Save each tree and its feature indices
        for (size_t i = 0; i < m_trees.size(); ++i) {
            // Save feature indices for this tree
            for (size_t j = 0; j < m_treeFeatures[i].size(); ++j) {
                file << m_treeFeatures[i][j];
                if (j < m_treeFeatures[i].size() - 1) {
                    file << ",";
                }
            }
            file << "\n";
            
            // Save tree to a temporary file and then append its content
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            m_trees[i]->save(tempFilename);
            
            std::ifstream treeFile(tempFilename);
            if (treeFile.is_open()) {
                file << treeFile.rdbuf();
                treeFile.close();
                std::remove(tempFilename.c_str());
            }
            
            file << "END_TREE\n";
        }
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Clear existing model
        m_trees.clear();
        m_treeFeatures.clear();
        
        std::string line;
        
        // Load hyperparameters
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_numTrees = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minSamplesSplit = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_maxFeatures = std::stod(cell);
            }
        }
        
        // Load number of trees
        size_t numTrees = 0;
        if (std::getline(file, line)) {
            numTrees = std::stoul(line);
        }
        
        // Load each tree
        for (size_t i = 0; i < numTrees; ++i) {
            // Load feature indices
            if (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                std::vector<size_t> featureIndices;
                
                while (std::getline(ss, cell, ',')) {
                    featureIndices.push_back(std::stoul(cell));
                }
                
                m_treeFeatures.push_back(featureIndices);
            }
            
            // Load tree to a temporary file
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            std::ofstream tempFile(tempFilename);
            
            while (std::getline(file, line) && line != "END_TREE") {
                tempFile << line << "\n";
            }
            
            tempFile.close();
            
            // Load tree from the temporary file
            auto tree = std::make_unique<DecisionTreeRegressor>();
            tree->load(tempFilename);
            m_trees.push_back(std::move(tree));
            
            // Remove temporary file
            std::remove(tempFilename.c_str());
        }
    }
    
private:
    size_t m_numTrees;
    size_t m_maxDepth;
    size_t m_minSamplesSplit;
    double m_maxFeatures;
    std::vector<std::unique_ptr<DecisionTreeRegressor>> m_trees;
    std::vector<std::vector<size_t>> m_treeFeatures;
    
    // Create bootstrap sample indices
    std::vector<size_t> bootstrapSample(size_t numSamples) const {
        std::vector<size_t> indices;
        indices.reserve(numSamples);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::uniform_int_distribution<size_t> dist(0, numSamples - 1);
        
        for (size_t i = 0; i < numSamples; ++i) {
            indices.push_back(dist(g));
        }
        
        return indices;
    }
    
    // Select feature indices for a tree
    std::vector<size_t> selectFeatures(size_t numFeatures, size_t numToSelect) const {
        std::vector<size_t> allIndices(numFeatures);
        for (size_t i = 0; i < numFeatures; ++i) {
            allIndices[i] = i;
        }
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(allIndices.begin(), allIndices.end(), g);
        
        return std::vector<size_t>(allIndices.begin(), allIndices.begin() + numToSelect);
    }
};

// XGBoost Implementation

// Tree structure for XGBoost
struct XGBTreeNode {
    bool isLeaf = false;
    size_t featureIndex = 0;
    double threshold = 0.0;
    double weight = 0.0;
    double gain = 0.0;
    std::unique_ptr<XGBTreeNode> leftChild;
    std::unique_ptr<XGBTreeNode> rightChild;
    
    // For tree serialization
    void serialize(std::ostream& os) const {
        os << (isLeaf ? 1 : 0) << ",";
        os << featureIndex << ",";
        os << threshold << ",";
        os << weight << ",";
        os << gain << "\n";
        
        if (!isLeaf) {
            if (leftChild) {
                leftChild->serialize(os);
            } else {
                os << "null\n";
            }
            
            if (rightChild) {
                rightChild->serialize(os);
            } else {
                os << "null\n";
            }
        }
    }
    
    static std::unique_ptr<XGBTreeNode> deserialize(std::istream& is) {
        std::string line;
        if (!std::getline(is, line) || line == "null") {
            return nullptr;
        }
        
        std::stringstream ss(line);
        std::string cell;
        auto node = std::make_unique<XGBTreeNode>();
        
        if (std::getline(ss, cell, ',')) {
            node->isLeaf = (std::stoi(cell) == 1);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->featureIndex = std::stoul(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->threshold = std::stod(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->weight = std::stod(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->gain = std::stod(cell);
        }
        
        if (!node->isLeaf) {
            node->leftChild = deserialize(is);
            node->rightChild = deserialize(is);
        }
        
        return node;
    }
};

// Single XGBoost tree
class XGBTree {
public:
    XGBTree(size_t maxDepth = 6, size_t minSamplesSplit = 2, double minLoss = 0.0, 
           double lambda = 1.0, double gamma = 0.0)
        : m_maxDepth(maxDepth), m_minSamplesSplit(minSamplesSplit), 
          m_minLoss(minLoss), m_lambda(lambda), m_gamma(gamma), m_root(nullptr) {}
    
    void fit(const Matrix& features, const std::vector<double>& gradients, 
             const std::vector<double>& hessians) {
        
        if (gradients.size() != features.rows() || hessians.size() != features.rows()) {
            throw std::invalid_argument("Gradients and hessians size mismatch");
        }
        
        // Build tree
        m_root = buildTree(features, gradients, hessians, 0);
    }
    
    std::vector<double> predict(const Matrix& features) const {
        std::vector<double> predictions(features.rows(), 0.0);
        
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions[i] = predictSample(features, i);
        }
        
        return predictions;
    }
    
    // Save tree to file
    void save(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_maxDepth << "," << m_minSamplesSplit << "," 
             << m_minLoss << "," << m_lambda << "," << m_gamma << "\n";
        
        // Save tree structure
        if (m_root) {
            m_root->serialize(file);
        } else {
            file << "null\n";
        }
    }
    
    // Load tree from file
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minSamplesSplit = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minLoss = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_lambda = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_gamma = std::stod(cell);
            }
        }
        
        // Load tree structure
        m_root = XGBTreeNode::deserialize(file);
    }
    
private:
    size_t m_maxDepth;
    size_t m_minSamplesSplit;
    double m_minLoss;  // Minimum loss reduction
    double m_lambda;   // L2 regularization
    double m_gamma;    // Minimum gain for node splitting
    std::unique_ptr<XGBTreeNode> m_root;
    
    double predictSample(const Matrix& features, size_t sampleIndex) const {
        const XGBTreeNode* node = m_root.get();
        while (node && !node->isLeaf) {
            if (features.at(sampleIndex, node->featureIndex) <= node->threshold) {
                node = node->leftChild.get();
            } else {
                node = node->rightChild.get();
            }
        }
        return node ? node->weight : 0.0;
    }
    
    // Calculate optimal weight for a node
    double calculateWeight(const std::vector<double>& gradients, 
                          const std::vector<double>& hessians) const {
        double sumGradients = 0.0;
        double sumHessians = 0.0;
        
        for (size_t i = 0; i < gradients.size(); ++i) {
            sumGradients += gradients[i];
            sumHessians += hessians[i];
        }
        
        return -sumGradients / (sumHessians + m_lambda);
    }
    
    // Calculate gain for a split
    double calculateSplitGain(const std::vector<double>& leftGradients, 
                            const std::vector<double>& leftHessians,
                            const std::vector<double>& rightGradients, 
                            const std::vector<double>& rightHessians,
                            const std::vector<double>& parentGradients, 
                            const std::vector<double>& parentHessians) const {
        
        double parentWeight = calculateWeight(parentGradients, parentHessians);
        double parentScore = -0.5 * parentWeight * parentWeight * std::accumulate(parentHessians.begin(), parentHessians.end(), 0.0);
        
        double leftWeight = calculateWeight(leftGradients, leftHessians);
        double leftScore = -0.5 * leftWeight * leftWeight * std::accumulate(leftHessians.begin(), leftHessians.end(), 0.0);
        
        double rightWeight = calculateWeight(rightGradients, rightHessians);
        double rightScore = -0.5 * rightWeight * rightWeight * std::accumulate(rightHessians.begin(), rightHessians.end(), 0.0);
        
        double gain = leftScore + rightScore - parentScore - m_gamma;
        return gain;
    }
    
    // Find best split for a node
    std::tuple<size_t, double, double> findBestSplit(
        const Matrix& features, 
        const std::vector<double>& gradients, 
        const std::vector<double>& hessians) const {
        
        size_t bestFeatureIndex = 0;
        double bestThreshold = 0.0;
        double bestGain = -std::numeric_limits<double>::max();
        
        for (size_t featureIndex = 0; featureIndex < features.cols(); ++featureIndex) {
            // Get unique values for the feature
            std::set<double> uniqueValues;
            for (size_t i = 0; i < features.rows(); ++i) {
                uniqueValues.insert(features.at(i, featureIndex));
            }
            
            // Try each unique value as threshold
            for (double threshold : uniqueValues) {
                // Split samples
                std::vector<double> leftGradients, leftHessians, rightGradients, rightHessians;
                
                for (size_t i = 0; i < features.rows(); ++i) {
                    if (features.at(i, featureIndex) <= threshold) {
                        leftGradients.push_back(gradients[i]);
                        leftHessians.push_back(hessians[i]);
                    } else {
                        rightGradients.push_back(gradients[i]);
                        rightHessians.push_back(hessians[i]);
                    }
                }
                
                // Skip if split doesn't meet minimum samples criteria
                if (leftGradients.size() < m_minSamplesSplit || rightGradients.size() < m_minSamplesSplit) {
                    continue;
                }
                
                // Calculate gain
                double gain = calculateSplitGain(leftGradients, leftHessians, rightGradients, rightHessians, 
                                             gradients, hessians);
                
                // Update best if this split is better
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }
        
        return {bestFeatureIndex, bestThreshold, bestGain};
    }
    
    // Recursively build tree
    std::unique_ptr<XGBTreeNode> buildTree(
        const Matrix& features, 
        const std::vector<double>& gradients, 
        const std::vector<double>& hessians, 
        size_t depth) {
        
        auto node = std::make_unique<XGBTreeNode>();
        
        // Calculate node weight
        double weight = calculateWeight(gradients, hessians);
        node->weight = weight;
        
        // Stop criteria
        if (depth >= m_maxDepth || gradients.size() <= m_minSamplesSplit) {
            node->isLeaf = true;
            return node;
        }
        
        // Find best split
        auto [featureIndex, threshold, gain] = findBestSplit(features, gradients, hessians);
        
        // If no good split is found or gain is too small, make a leaf node
        if (gain <= m_minLoss) {
            node->isLeaf = true;
            return node;
        }
        
        // Set node parameters
        node->isLeaf = false;
        node->featureIndex = featureIndex;
        node->threshold = threshold;
        node->gain = gain;
        
        // Split samples
        std::vector<size_t> leftIndices, rightIndices;
        for (size_t i = 0; i < features.rows(); ++i) {
            if (features.at(i, featureIndex) <= threshold) {
                leftIndices.push_back(i);
            } else {
                rightIndices.push_back(i);
            }
        }
        
        // Create left and right data
        Matrix leftFeatures(leftIndices.size(), features.cols());
        std::vector<double> leftGradients(leftIndices.size());
        std::vector<double> leftHessians(leftIndices.size());
        
        Matrix rightFeatures(rightIndices.size(), features.cols());
        std::vector<double> rightGradients(rightIndices.size());
        std::vector<double> rightHessians(rightIndices.size());
        
        for (size_t i = 0; i < leftIndices.size(); ++i) {
            size_t idx = leftIndices[i];
            leftGradients[i] = gradients[idx];
            leftHessians[i] = hessians[idx];
            
            for (size_t j = 0; j < features.cols(); ++j) {
                leftFeatures.at(i, j) = features.at(idx, j);
            }
        }
        
        for (size_t i = 0; i < rightIndices.size(); ++i) {
            size_t idx = rightIndices[i];
            rightGradients[i] = gradients[idx];
            rightHessians[i] = hessians[idx];
            
            for (size_t j = 0; j < features.cols(); ++j) {
                rightFeatures.at(i, j) = features.at(idx, j);
            }
        }
        
        // Recursively build subtrees
        node->leftChild = buildTree(leftFeatures, leftGradients, leftHessians, depth + 1);
        node->rightChild = buildTree(rightFeatures, rightGradients, rightHessians, depth + 1);
        
        return node;
    }
};

// XGBoost implementation
class XGBoost : public Model {
public:
    XGBoost(size_t numTrees = 100, size_t maxDepth = 6, double learningRate = 0.3,
           double lambda = 1.0, double gamma = 0.0)
        : m_numTrees(numTrees), m_maxDepth(maxDepth), m_learningRate(learningRate),
          m_lambda(lambda), m_gamma(gamma), m_initialPrediction(0.0) {}
    
    void fit(const DataSet& dataset, size_t epochs = 1, size_t = 32) override {
        const Matrix& features = dataset.features();
        const Matrix& targets = dataset.labels();
        
        if (targets.cols() != 1) {
            throw std::invalid_argument("XGBoost currently only supports single target regression");
        }
        
        // Clear existing trees
        m_trees.clear();
        
        // Initialize predictions with base prediction (mean of targets)
        double sum = 0.0;
        for (size_t i = 0; i < targets.rows(); ++i) {
            sum += targets.at(i, 0);
        }
        m_initialPrediction = sum / targets.rows();
        
        // Current predictions for each sample
        std::vector<double> predictions(features.rows(), m_initialPrediction);
        
        // Build trees iteratively
        for (size_t i = 0; i < m_numTrees * epochs; ++i) {
            // Calculate gradients and hessians for current predictions
            std::vector<double> gradients(features.rows());
            std::vector<double> hessians(features.rows());
            
            // For MSE objective: grad = pred - target, hess = 1.0
            for (size_t j = 0; j < features.rows(); ++j) {
                gradients[j] = predictions[j] - targets.at(j, 0);
                hessians[j] = 1.0;
            }
            
            // Build new tree
            auto tree = std::make_unique<XGBTree>(m_maxDepth, 2, 0.0, m_lambda, m_gamma);
            tree->fit(features, gradients, hessians);
            
            // Update predictions
            std::vector<double> treePredictions = tree->predict(features);
            for (size_t j = 0; j < predictions.size(); ++j) {
                predictions[j] -= m_learningRate * treePredictions[j];
            }
            
            // Store tree
            m_trees.push_back(std::move(tree));
            
            if ((i + 1) % 10 == 0 || i == 0 || i == m_numTrees * epochs - 1) {
                // Calculate MSE
                double mse = 0.0;
                for (size_t j = 0; j < predictions.size(); ++j) {
                    double error = predictions[j] - targets.at(j, 0);
                    mse += error * error;
                }
                mse /= predictions.size();
                
                std::cout << "Trained tree " << (i + 1) << "/" << (m_numTrees * epochs) 
                          << ", MSE: " << mse << std::endl;
            }
        }
    }
    
    Matrix predict(const Matrix& features) const override {
        Matrix predictions(features.rows(), 1);
        
        // Initialize with base prediction
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions.at(i, 0) = m_initialPrediction;
        }
        
        // Add contributions from each tree
        for (const auto& tree : m_trees) {
            std::vector<double> treePredictions = tree->predict(features);
            for (size_t i = 0; i < features.rows(); ++i) {
                predictions.at(i, 0) -= m_learningRate * treePredictions[i];
            }
        }
        
        return predictions;
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_numTrees << "," << m_maxDepth << "," << m_learningRate << ","
             << m_lambda << "," << m_gamma << "," << m_initialPrediction << "\n";
        
        // Save number of trees
        file << m_trees.size() << "\n";
        
        // Save each tree
        for (size_t i = 0; i < m_trees.size(); ++i) {
            // Save tree to a temporary file and then append its content
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            m_trees[i]->save(tempFilename);
            
            std::ifstream treeFile(tempFilename);
            if (treeFile.is_open()) {
                file << treeFile.rdbuf();
                treeFile.close();
                std::remove(tempFilename.c_str());
            }
            
            file << "END_TREE\n";
        }
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Clear existing model
        m_trees.clear();
        
        std::string line;
        
        // Load hyperparameters
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_numTrees = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_learningRate = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_lambda = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_gamma = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_initialPrediction = std::stod(cell);
            }
        }
        
        // Load number of trees
        size_t numTrees = 0;
        if (std::getline(file, line)) {
            numTrees = std::stoul(line);
        }
        
        // Load each tree
        for (size_t i = 0; i < numTrees; ++i) {
            // Load tree to a temporary file
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            std::ofstream tempFile(tempFilename);
            
            while (std::getline(file, line) && line != "END_TREE") {
                tempFile << line << "\n";
            }
            
            tempFile.close();
            
            // Load tree from the temporary file
            auto tree = std::make_unique<XGBTree>();
            tree->load(tempFilename);
            m_trees.push_back(std::move(tree));
            
            // Remove temporary file
            std::remove(tempFilename.c_str());
        }
    }
    
private:
    size_t m_numTrees;
    size_t m_maxDepth;
    double m_learningRate;
    double m_lambda;  // L2 regularization
    double m_gamma;   // Minimum gain for node splitting
    double m_initialPrediction;
    std::vector<std::unique_ptr<XGBTree>> m_trees;
};

// ------------------- LightGBM Implementation -------------------

// Simple histogram structure for LightGBM
struct Histogram {
    std::vector<double> gradSum;    // Sum of gradients in bin
    std::vector<double> hessSum;    // Sum of hessians in bin
    std::vector<size_t> count;      // Count of samples in bin
    
    Histogram(size_t numBins = 0) {
        if (numBins > 0) {
            gradSum.resize(numBins, 0.0);
            hessSum.resize(numBins, 0.0);
            count.resize(numBins, 0);
        }
    }
    
    void add(size_t bin, double grad, double hess) {
        if (bin < gradSum.size()) {
            gradSum[bin] += grad;
            hessSum[bin] += hess;
            count[bin]++;
        }
    }
    
    void clear() {
        std::fill(gradSum.begin(), gradSum.end(), 0.0);
        std::fill(hessSum.begin(), hessSum.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
    }
};

// Tree node for LightGBM
struct LGBMTreeNode {
    bool isLeaf = false;
    size_t featureIndex = 0;
    size_t bin = 0;           // Bin index for the threshold
    double threshold = 0.0;   // Actual threshold value
    double weight = 0.0;
    double gain = 0.0;
    std::unique_ptr<LGBMTreeNode> leftChild;
    std::unique_ptr<LGBMTreeNode> rightChild;
    
    // For tree serialization
    void serialize(std::ostream& os) const {
        os << (isLeaf ? 1 : 0) << ",";
        os << featureIndex << ",";
        os << bin << ",";
        os << threshold << ",";
        os << weight << ",";
        os << gain << "\n";
        
        if (!isLeaf) {
            if (leftChild) {
                leftChild->serialize(os);
            } else {
                os << "null\n";
            }
            
            if (rightChild) {
                rightChild->serialize(os);
            } else {
                os << "null\n";
            }
        }
    }
    
    static std::unique_ptr<LGBMTreeNode> deserialize(std::istream& is) {
        std::string line;
        if (!std::getline(is, line) || line == "null") {
            return nullptr;
        }
        
        std::stringstream ss(line);
        std::string cell;
        auto node = std::make_unique<LGBMTreeNode>();
        
        if (std::getline(ss, cell, ',')) {
            node->isLeaf = (std::stoi(cell) == 1);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->featureIndex = std::stoul(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->bin = std::stoul(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->threshold = std::stod(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->weight = std::stod(cell);
        }
        
        if (std::getline(ss, cell, ',')) {
            node->gain = std::stod(cell);
        }
        
        if (!node->isLeaf) {
            node->leftChild = deserialize(is);
            node->rightChild = deserialize(is);
        }
        
        return node;
    }
};

// LightGBM tree
class LGBMTree {
public:
    LGBMTree(size_t maxDepth = 6, size_t minSamplesSplit = 20, double minLoss = 0.0, 
            double lambda = 1.0, double gamma = 0.0, size_t numBins = 255)
        : m_maxDepth(maxDepth), m_minSamplesSplit(minSamplesSplit), 
          m_minLoss(minLoss), m_lambda(lambda), m_gamma(gamma), 
          m_numBins(numBins), m_root(nullptr) {}
    
    void fit(const Matrix& features, const std::vector<double>& gradients, 
             const std::vector<double>& hessians) {
        
        if (gradients.size() != features.rows() || hessians.size() != features.rows()) {
            throw std::invalid_argument("Gradients and hessians size mismatch");
        }
        
        // Create feature bins
        createBins(features);
        
        // Build tree
        m_root = buildTree(features, gradients, hessians, 0);
    }
    
    std::vector<double> predict(const Matrix& features) const {
        std::vector<double> predictions(features.rows(), 0.0);
        
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions[i] = predictSample(features, i);
        }
        
        return predictions;
    }
    
    // Save tree to file
    void save(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_maxDepth << "," << m_minSamplesSplit << "," 
             << m_minLoss << "," << m_lambda << "," << m_gamma << "," << m_numBins << "\n";
        
        // Save feature bins
        file << m_featureBins.size() << "\n";
        for (const auto& bins : m_featureBins) {
            file << bins.size();
            for (double bin : bins) {
                file << "," << bin;
            }
            file << "\n";
        }
        
        // Save tree structure
        if (m_root) {
            m_root->serialize(file);
        } else {
            file << "null\n";
        }
    }
    
    // Load tree from file
    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        std::string line;
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minSamplesSplit = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_minLoss = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_lambda = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_gamma = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_numBins = std::stoul(cell);
            }
        }
        
        // Load feature bins
        if (std::getline(file, line)) {
            size_t numFeatures = std::stoul(line);
            m_featureBins.resize(numFeatures);
            
            for (size_t i = 0; i < numFeatures; ++i) {
                if (std::getline(file, line)) {
                    std::stringstream ss(line);
                    std::string cell;
                    
                    if (std::getline(ss, cell, ',')) {
                        size_t numBins = std::stoul(cell);
                        m_featureBins[i].resize(numBins);
                        
                        for (size_t j = 0; j < numBins; ++j) {
                            if (std::getline(ss, cell, ',')) {
                                m_featureBins[i][j] = std::stod(cell);
                            }
                        }
                    }
                }
            }
        }
        
        // Load tree structure
        m_root = LGBMTreeNode::deserialize(file);
    }
    
private:
    size_t m_maxDepth;
    size_t m_minSamplesSplit;
    double m_minLoss;  // Minimum loss reduction
    double m_lambda;   // L2 regularization
    double m_gamma;    // Minimum gain for node splitting
    size_t m_numBins;  // Number of histogram bins
    std::unique_ptr<LGBMTreeNode> m_root;
    std::vector<std::vector<double>> m_featureBins;  // Bin boundaries for each feature
    
    void createBins(const Matrix& features) {
        m_featureBins.clear();
        m_featureBins.resize(features.cols());
        
        for (size_t j = 0; j < features.cols(); ++j) {
            // Extract values for this feature
            std::vector<double> values(features.rows());
            for (size_t i = 0; i < features.rows(); ++i) {
                values[i] = features.at(i, j);
            }
            
            // Sort values
            std::sort(values.begin(), values.end());
            
            // Remove duplicates
            auto last = std::unique(values.begin(), values.end());
            values.erase(last, values.end());
            
            // If fewer unique values than bins, use those
            if (values.size() <= m_numBins) {
                m_featureBins[j] = values;
            } else {
                // Otherwise, create equally spaced quantile bins
                m_featureBins[j].resize(m_numBins);
                for (size_t binIdx = 0; binIdx < m_numBins; ++binIdx) {
                    size_t idx = binIdx * values.size() / m_numBins;
                    m_featureBins[j][binIdx] = values[idx];
                }
            }
        }
    }
    
    size_t getBin(double value, size_t featureIndex) const {
        // Find the bin for a feature value using binary search
        const auto& bins = m_featureBins[featureIndex];
        auto it = std::upper_bound(bins.begin(), bins.end(), value);
        return std::distance(bins.begin(), it);
    }
    
    double predictSample(const Matrix& features, size_t sampleIndex) const {
        const LGBMTreeNode* node = m_root.get();
        while (node && !node->isLeaf) {
            if (features.at(sampleIndex, node->featureIndex) <= node->threshold) {
                node = node->leftChild.get();
            } else {
                node = node->rightChild.get();
            }
        }
        return node ? node->weight : 0.0;
    }
    
    // Calculate optimal weight for a node
    double calculateWeight(const std::vector<double>& gradients, 
                          const std::vector<double>& hessians) const {
        double sumGradients = 0.0;
        double sumHessians = 0.0;
        
        for (size_t i = 0; i < gradients.size(); ++i) {
            sumGradients += gradients[i];
            sumHessians += hessians[i];
        }
        
        return -sumGradients / (sumHessians + m_lambda);
    }
    
    // Find best split using histogram-based approach
    std::tuple<size_t, size_t, double, double> findBestSplit(
        const Matrix& features, 
        const std::vector<double>& gradients, 
        const std::vector<double>& hessians) const {
        
        size_t bestFeatureIndex = 0;
        size_t bestBin = 0;
        double bestGain = -std::numeric_limits<double>::max();
        double bestThreshold = 0.0;
        
        // Parent node stats
        double parentGradSum = 0.0;
        double parentHessSum = 0.0;
        for (size_t i = 0; i < gradients.size(); ++i) {
            parentGradSum += gradients[i];
            parentHessSum += hessians[i];
        }
        
        // Try each feature
        for (size_t featureIndex = 0; featureIndex < features.cols(); ++featureIndex) {
            // Build histogram for this feature
            Histogram hist(m_featureBins[featureIndex].size() + 1);  // +1 for overflow bin
            
            for (size_t i = 0; i < features.rows(); ++i) {
                size_t bin = getBin(features.at(i, featureIndex), featureIndex);
                hist.add(bin, gradients[i], hessians[i]);
            }
            
            // Try each bin as split point
            double leftGradSum = 0.0;
            double leftHessSum = 0.0;
            double rightGradSum = parentGradSum;
            double rightHessSum = parentHessSum;
            
            for (size_t bin = 0; bin < hist.gradSum.size() - 1; ++bin) {
                // Skip if no samples in this bin
                if (hist.count[bin] == 0) {
                    continue;
                }
                
                // Move this bin from right to left
                leftGradSum += hist.gradSum[bin];
                leftHessSum += hist.hessSum[bin];
                rightGradSum -= hist.gradSum[bin];
                rightHessSum -= hist.hessSum[bin];
                
                // Skip if either side doesn't have enough samples
                if (leftHessSum < m_minSamplesSplit || rightHessSum < m_minSamplesSplit) {
                    continue;
                }
                
                // Calculate gain
                double leftWeight = -leftGradSum / (leftHessSum + m_lambda);
                double rightWeight = -rightGradSum / (rightHessSum + m_lambda);
                
                double gain = 0.5 * (
                    (leftGradSum * leftGradSum) / (leftHessSum + m_lambda) +
                    (rightGradSum * rightGradSum) / (rightHessSum + m_lambda) -
                    (parentGradSum * parentGradSum) / (parentHessSum + m_lambda)
                ) - m_gamma;
                
                // Update best if this split is better
                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeatureIndex = featureIndex;
                    bestBin = bin;
                    bestThreshold = m_featureBins[featureIndex][bin];
                }
            }
        }
        
        return {bestFeatureIndex, bestBin, bestThreshold, bestGain};
    }
    
    // Recursively build tree
    std::unique_ptr<LGBMTreeNode> buildTree(
        const Matrix& features, 
        const std::vector<double>& gradients, 
        const std::vector<double>& hessians, 
        size_t depth) {
        
        auto node = std::make_unique<LGBMTreeNode>();
        
        // Calculate node weight
        double weight = calculateWeight(gradients, hessians);
        node->weight = weight;
        
        // Stop criteria
        if (depth >= m_maxDepth || gradients.size() <= m_minSamplesSplit) {
            node->isLeaf = true;
            return node;
        }
        
        // Find best split
        auto [featureIndex, bin, threshold, gain] = findBestSplit(features, gradients, hessians);
        
        // If no good split is found or gain is too small, make a leaf node
        if (gain <= m_minLoss) {
            node->isLeaf = true;
            return node;
        }
        
        // Set node parameters
        node->isLeaf = false;
        node->featureIndex = featureIndex;
        node->bin = bin;
        node->threshold = threshold;
        node->gain = gain;
        
        // Split samples
        std::vector<size_t> leftIndices, rightIndices;
        for (size_t i = 0; i < features.rows(); ++i) {
            if (features.at(i, featureIndex) <= threshold) {
                leftIndices.push_back(i);
            } else {
                rightIndices.push_back(i);
            }
        }
        
        // Create left and right data
        Matrix leftFeatures(leftIndices.size(), features.cols());
        std::vector<double> leftGradients(leftIndices.size());
        std::vector<double> leftHessians(leftIndices.size());
        
        Matrix rightFeatures(rightIndices.size(), features.cols());
        std::vector<double> rightGradients(rightIndices.size());
        std::vector<double> rightHessians(rightIndices.size());
        
        for (size_t i = 0; i < leftIndices.size(); ++i) {
            size_t idx = leftIndices[i];
            leftGradients[i] = gradients[idx];
            leftHessians[i] = hessians[idx];
            
            for (size_t j = 0; j < features.cols(); ++j) {
                leftFeatures.at(i, j) = features.at(idx, j);
            }
        }
        
        for (size_t i = 0; i < rightIndices.size(); ++i) {
            size_t idx = rightIndices[i];
            rightGradients[i] = gradients[idx];
            rightHessians[i] = hessians[idx];
            
            for (size_t j = 0; j < features.cols(); ++j) {
                rightFeatures.at(i, j) = features.at(idx, j);
            }
        }
        
        // Recursively build subtrees
        node->leftChild = buildTree(leftFeatures, leftGradients, leftHessians, depth + 1);
        node->rightChild = buildTree(rightFeatures, rightGradients, rightHessians, depth + 1);
        
        return node;
    }
};

// LightGBM implementation
class LightGBM : public Model {
public:
    LightGBM(size_t numTrees = 100, size_t maxDepth = 6, double learningRate = 0.1,
            double lambda = 1.0, double gamma = 0.0, size_t numBins = 255)
        : m_numTrees(numTrees), m_maxDepth(maxDepth), m_learningRate(learningRate),
          m_lambda(lambda), m_gamma(gamma), m_numBins(numBins), m_initialPrediction(0.0) {}
    
    void fit(const DataSet& dataset, size_t epochs = 1, size_t = 32) override {
        const Matrix& features = dataset.features();
        const Matrix& targets = dataset.labels();
        
        if (targets.cols() != 1) {
            throw std::invalid_argument("LightGBM currently only supports single target regression");
        }
        
        // Clear existing trees
        m_trees.clear();
        
        // Initialize predictions with base prediction (mean of targets)
        double sum = 0.0;
        for (size_t i = 0; i < targets.rows(); ++i) {
            sum += targets.at(i, 0);
        }
        m_initialPrediction = sum / targets.rows();
        
        // Current predictions for each sample
        std::vector<double> predictions(features.rows(), m_initialPrediction);
        
        // Build trees iteratively
        for (size_t i = 0; i < m_numTrees * epochs; ++i) {
            // Calculate gradients and hessians for current predictions
            std::vector<double> gradients(features.rows());
            std::vector<double> hessians(features.rows());
            
            // For MSE objective: grad = pred - target, hess = 1.0
            for (size_t j = 0; j < features.rows(); ++j) {
                gradients[j] = predictions[j] - targets.at(j, 0);
                hessians[j] = 1.0;
            }
            
            // Build new tree
            auto tree = std::make_unique<LGBMTree>(
                m_maxDepth, 20, 0.0, m_lambda, m_gamma, m_numBins);
            tree->fit(features, gradients, hessians);
            
            // Update predictions
            std::vector<double> treePredictions = tree->predict(features);
            for (size_t j = 0; j < predictions.size(); ++j) {
                predictions[j] -= m_learningRate * treePredictions[j];
            }
            
            // Store tree
            m_trees.push_back(std::move(tree));
            
            if ((i + 1) % 10 == 0 || i == 0 || i == m_numTrees * epochs - 1) {
                // Calculate MSE
                double mse = 0.0;
                for (size_t j = 0; j < predictions.size(); ++j) {
                    double error = predictions[j] - targets.at(j, 0);
                    mse += error * error;
                }
                mse /= predictions.size();
                
                std::cout << "Trained tree " << (i + 1) << "/" << (m_numTrees * epochs) 
                          << ", MSE: " << mse << std::endl;
            }
        }
    }
    
    Matrix predict(const Matrix& features) const override {
        Matrix predictions(features.rows(), 1);
        
        // Initialize with base prediction
        for (size_t i = 0; i < features.rows(); ++i) {
            predictions.at(i, 0) = m_initialPrediction;
        }
        
        // Add contributions from each tree
        for (const auto& tree : m_trees) {
            std::vector<double> treePredictions = tree->predict(features);
            for (size_t i = 0; i < features.rows(); ++i) {
                predictions.at(i, 0) -= m_learningRate * treePredictions[i];
            }
        }
        
        return predictions;
    }
    
    void saveModel(const std::string& filename) const override {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        // Save hyperparameters
        file << m_numTrees << "," << m_maxDepth << "," << m_learningRate << ","
             << m_lambda << "," << m_gamma << "," << m_numBins << "," << m_initialPrediction << "\n";
        
        // Save number of trees
        file << m_trees.size() << "\n";
        
        // Save each tree
        for (size_t i = 0; i < m_trees.size(); ++i) {
            // Save tree to a temporary file and then append its content
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            m_trees[i]->save(tempFilename);
            
            std::ifstream treeFile(tempFilename);
            if (treeFile.is_open()) {
                file << treeFile.rdbuf();
                treeFile.close();
                std::remove(tempFilename.c_str());
            }
            
            file << "END_TREE\n";
        }
    }
    
    void loadModel(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for reading: " + filename);
        }
        
        // Clear existing model
        m_trees.clear();
        
        std::string line;
        
        // Load hyperparameters
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            
            if (std::getline(ss, cell, ',')) {
                m_numTrees = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_maxDepth = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_learningRate = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_lambda = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_gamma = std::stod(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_numBins = std::stoul(cell);
            }
            
            if (std::getline(ss, cell, ',')) {
                m_initialPrediction = std::stod(cell);
            }
        }
        
        // Load number of trees
        size_t numTrees = 0;
        if (std::getline(file, line)) {
            numTrees = std::stoul(line);
        }
        
        // Load each tree
        for (size_t i = 0; i < numTrees; ++i) {
            // Load tree to a temporary file
            std::string tempFilename = filename + ".tree" + std::to_string(i);
            std::ofstream tempFile(tempFilename);
            
            while (std::getline(file, line) && line != "END_TREE") {
                tempFile << line << "\n";
            }
            
            tempFile.close();
            
            // Load tree from the temporary file
            auto tree = std::make_unique<LGBMTree>();
            tree->load(tempFilename);
            m_trees.push_back(std::move(tree));
            
            // Remove temporary file
            std::remove(tempFilename.c_str());
        }
    }
    
private:
    size_t m_numTrees;
    size_t m_maxDepth;
    double m_learningRate;
    double m_lambda;    // L2 regularization
    double m_gamma;     // Minimum gain for node splitting
    size_t m_numBins;   // Number of histogram bins
    double m_initialPrediction;
    std::vector<std::unique_ptr<LGBMTree>> m_trees;
};

} // namespace tbml

// Example usage
int main() {
    // Create a synthetic dataset for regression
    std::vector<std::vector<double>> features = {
        {1.0, 2.0, 3.0}, {2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}, {4.0, 5.0, 6.0},
        {5.0, 6.0, 7.0}, {6.0, 7.0, 8.0}, {7.0, 8.0, 9.0}, {8.0, 9.0, 10.0},
        {1.5, 2.5, 3.5}, {2.5, 3.5, 4.5}, {3.5, 4.5, 5.5}, {4.5, 5.5, 6.5},
        {5.5, 6.5, 7.5}, {6.5, 7.5, 8.5}, {7.5, 8.5, 9.5}, {8.5, 9.5, 10.5}
    };
    
    std::vector<std::vector<double>> regressionTargets = {
        {6.0}, {9.0}, {12.0}, {15.0}, {18.0}, {21.0}, {24.0}, {27.0},
        {7.5}, {10.5}, {13.5}, {16.5}, {19.5}, {22.5}, {25.5}, {28.5}
    };
    
    std::vector<std::vector<double>> classificationTargets = {
        {1.0}, {1.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0},
        {1.0}, {1.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0}
    };
    
    tbml::DataSet regressionData(features, regressionTargets);
    tbml::DataSet classificationData(features, classificationTargets);
    
    // Split the data
    auto [regressionTrain, regressionTest] = regressionData.trainTestSplit(0.25);
    auto [classificationTrain, classificationTest] = classificationData.trainTestSplit(0.25);
    
    // Test Linear Regression
    std::cout << "\n===== Linear Regression =====\n";
    auto linearRegression = std::make_shared<tbml::LinearRegression>(3);
    linearRegression->fit(regressionTrain, 100);
    double regressionMSE = linearRegression->evaluate(regressionTest);
    std::cout << "MSE on test data: " << regressionMSE << std::endl;
    
    // Test Logistic Regression
    /*std::cout << "\n===== Logistic Regression =====\n";
    auto logisticRegression = std::make_shared<tbml::LogisticRegression>(3, 1);
    logisticRegression->fit(classificationTrain, 100);
    double classificationAccuracy = logisticRegression->accuracy(classificationTest);
    std::cout << "Accuracy on test data: " << classificationAccuracy << std::endl;*/
    
    // Test Random Forest
    std::cout << "\n===== Random Forest =====\n";
    auto randomForest = std::make_shared<tbml::RandomForestRegressor>(10, 5);
    randomForest->fit(regressionTrain, 1);
    double rfMSE = randomForest->evaluate(regressionTest);
    std::cout << "Random Forest MSE on test data: " << rfMSE << std::endl;

    // Test XGBoost
    std::cout << "\n===== XGBoost =====\n";
    auto xgboost = std::make_shared<tbml::XGBoost>(20, 3, 0.1);  // 20 trees, max depth 3, learning rate 0.1
    xgboost->fit(regressionTrain, 1);
    double xgbMSE = xgboost->evaluate(regressionTest);
    std::cout << "XGBoost MSE on test data: " << xgbMSE << std::endl;
    
    // Test LightGBM
    std::cout << "\n===== LightGBM =====\n";
    auto lightgbm = std::make_shared<tbml::LightGBM>(20, 3, 0.1, 1.0, 0.0, 32);  // 20 trees, max depth 3, learning rate 0.1, lambda 1.0, gamma 0.0, 32 bins
    lightgbm->fit(regressionTrain, 1);
    double lgbmMSE = lightgbm->evaluate(regressionTest);
    std::cout << "LightGBM MSE on test data: " << lgbmMSE << std::endl;
    
    // Compare all models
    std::cout << "\n===== Model Comparison =====\n";
    std::cout << "Linear Regression MSE: " << regressionMSE << std::endl;
    std::cout << "Random Forest MSE: " << rfMSE << std::endl;
    std::cout << "XGBoost MSE: " << xgbMSE << std::endl;
    std::cout << "LightGBM MSE: " << lgbmMSE << std::endl;
    
    // Optional: Save and load model test
    std::cout << "\n===== Save and Load Test (XGBoost) =====\n";
    xgboost->saveModel("xgboost_model.txt");
    auto loadedXgboost = std::make_shared<tbml::XGBoost>();
    loadedXgboost->loadModel("xgboost_model.txt");
    double loadedXgbMSE = loadedXgboost->evaluate(regressionTest);
    std::cout << "Loaded XGBoost MSE on test data: " << loadedXgbMSE << std::endl;
    
    return 0;
}