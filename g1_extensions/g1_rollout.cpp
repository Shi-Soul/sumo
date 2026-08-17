#include "g1_rollout.h"
#include <pybind11/stl.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <chrono>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace py = pybind11;

// ONNX Policy wrapper class
class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::shared_ptr<Ort::Session>& session)
        : session_(session), memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        Ort::Allocator allocator(*session_, memory_info_);
        input_name_  = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        input_shape_  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        output_shape_ = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_size_ = static_cast<int>(input_shape_[1]);
        output_size_ = static_cast<int>(output_shape_[1]);
    }

    std::vector<float> run(const std::vector<float>& observation) {
        if ((int)observation.size() != input_size_) {
            throw std::runtime_error("Observation size does not match ONNX input dimension");
        }
        std::array<int64_t, 2> ishape = { 1, static_cast<int64_t>(observation.size()) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, const_cast<float*>(observation.data()), observation.size(), ishape.data(), 2);
        const char* in_names[1] = { input_name_.c_str() };
        const char* out_names[1] = { output_name_.c_str() };
        auto outputs = session_->Run(run_options_, in_names, &input_tensor, 1, out_names, 1);
        auto& out = outputs[0];
        float* ptr = out.GetTensorMutableData<float>();
        auto info = out.GetTensorTypeAndShapeInfo();
        size_t n = info.GetElementCount();
        return std::vector<float>(ptr, ptr + n);
    }

    int input_size() const { return input_size_; }
    int output_size() const { return output_size_; }

private:
    std::shared_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    Ort::RunOptions run_options_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    int input_size_ = 0;
    int output_size_ = 0;
};

// Utility functions
py::array_t<double> make_array_owned_g1(std::vector<double>& buf, int B, int T, int D) {
    std::vector<ssize_t> shape   = { B, T, D };
    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(double) * T * D),
        static_cast<ssize_t>(sizeof(double) *     D),
        static_cast<ssize_t>(sizeof(double))
    };
    auto heap_buf = new std::vector<double>(std::move(buf));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<double>*>(p); });
    return py::array_t<double>(shape, strides, heap_buf->data(), free_when_done);
}

// ONNX session allocator
static std::shared_ptr<Ort::Session> allocate_shared_session(const std::string& onnx_path) {
    static Ort::Env env;
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    return std::make_shared<Ort::Session>(env, onnx_path.c_str(), opts);
}

// Metadata structure for G1 policy
struct G1Metadata {
    int num_joints;
    int obs_dim;
    std::vector<double> default_joint_pos;
    std::vector<double> action_scale;
    std::vector<std::string> joint_names;
    std::vector<int> arm_joint_indices;

    G1Metadata() : num_joints(29), obs_dim(99) {}
};

// Global metadata storage
static G1Metadata g_g1_metadata;
static bool g_g1_metadata_loaded = false;

// Helper function to parse comma-separated values
static std::vector<double> parse_csv_floats(const std::string& csv) {
    std::vector<double> result;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(item));
    }
    return result;
}

static std::vector<std::string> parse_csv_strings(const std::string& csv) {
    std::vector<std::string> result;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(item);
    }
    return result;
}

// Parse metadata from ONNX model using ONNX Runtime session
static void parse_g1_metadata(const std::shared_ptr<Ort::Session>& session) {
    if (g_g1_metadata_loaded) return;

    try {
        // Get model metadata from session
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::ModelMetadata metadata = session->GetModelMetadata();

        // Parse metadata properties
        std::unordered_map<std::string, std::string> metadata_map;

        // Try to get custom metadata keys
        std::vector<std::string> keys_to_try = {
            "default_joint_pos", "action_scale", "joint_names"
        };

        for (const auto& key : keys_to_try) {
            try {
                Ort::AllocatedStringPtr value_ptr = metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
                if (value_ptr) {
                    metadata_map[key] = std::string(value_ptr.get());
                }
            } catch (...) {
                // Key not found, skip
            }
        }

        // Parse default_joint_pos
        if (metadata_map.count("default_joint_pos")) {
            g_g1_metadata.default_joint_pos = parse_csv_floats(metadata_map["default_joint_pos"]);
            g_g1_metadata.num_joints = static_cast<int>(g_g1_metadata.default_joint_pos.size());
        } else {
            // Fallback to zeros
            g_g1_metadata.default_joint_pos.assign(29, 0.0);
        }

        // Parse action_scale
        if (metadata_map.count("action_scale")) {
            g_g1_metadata.action_scale = parse_csv_floats(metadata_map["action_scale"]);
        } else {
            // Fallback to 0.25 for all joints
            g_g1_metadata.action_scale.assign(g_g1_metadata.num_joints, 0.25);
        }

        // Parse joint_names
        if (metadata_map.count("joint_names")) {
            g_g1_metadata.joint_names = parse_csv_strings(metadata_map["joint_names"]);
        }

    } catch (const std::exception& e) {
        // If metadata parsing fails, use defaults
        g_g1_metadata.default_joint_pos.assign(29, 0.0);
        g_g1_metadata.action_scale.assign(29, 0.25);
        g_g1_metadata.num_joints = 29;
    }

    // Arm joint indices (15-28) - hard-coded as in simulate_g1.py
    g_g1_metadata.arm_joint_indices = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

    // Calculate observation dimension
    g_g1_metadata.obs_dim = 3 + 3 + 3 + g_g1_metadata.num_joints * 3 + 3;

    g_g1_metadata_loaded = true;
}

static std::string get_g1_policy_path() {
    // Check G1_EXTENSIONS_POLICY_DIR env var first (set by g1_extensions/__init__.py)
    const char* policy_dir = std::getenv("G1_EXTENSIONS_POLICY_DIR");
    if (policy_dir) {
        return std::string(policy_dir) + "/g1_velocity_policy.onnx";
    }
    // Fallback to relative path
    return std::string("g1_extensions/policy/g1_velocity_policy.onnx");
}

// =============================================================================
// G1ThreadPool Implementation
// =============================================================================

G1ThreadPool::G1ThreadPool(int num_threads)
    : num_threads_(num_threads), stop_(false), active_workers_(0), total_tasks_(0), completed_tasks_(0) {
    if (num_threads_ > 0) {
        threads_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; ++i) {
            threads_.emplace_back(&G1ThreadPool::worker_thread, this);
        }
    }
}

G1ThreadPool::~G1ThreadPool() {
    if (num_threads_ > 0) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread &worker : threads_) {
            worker.join();
        }
    }
}

void G1ThreadPool::execute_parallel(std::function<void(int)> func, int total_work) {
    if (num_threads_ == 0) {
        // Single-threaded execution
        for (int i = 0; i < total_work; ++i) {
            func(i);
        }
    } else {
        // Multi-threaded execution
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            total_tasks_ = total_work;
            completed_tasks_ = 0;

            for (int i = 0; i < total_work; ++i) {
                tasks_.push([func, i]() { func(i); });
            }
        }
        condition_.notify_all();

        std::unique_lock<std::mutex> lock(queue_mutex_);
        finished_.wait(lock, [this]() { return completed_tasks_ == total_tasks_; });
    }
}

void G1ThreadPool::worker_thread() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty())
                return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            completed_tasks_++;
            if (completed_tasks_ == total_tasks_) {
                finished_.notify_one();
            }
        }
    }
}

// =============================================================================
// G1Rollout Implementation
// =============================================================================

G1Rollout::G1Rollout(int nthread, double cutoff_time) : num_threads_(nthread), cutoff_time_(cutoff_time) {
    initialize_policy();
    if (num_threads_ != 0) {
        thread_pool_ = std::make_unique<G1ThreadPool>(num_threads_);
    }
}

G1Rollout::~G1Rollout() {
    close();
}

void G1Rollout::close() {
    if (!closed_) {
        thread_pool_.reset();
        policy_.reset();
        onnx_session_.reset();
        closed_ = true;
    }
}

void G1Rollout::initialize_policy() {
    std::string policy_path = get_g1_policy_path();
    onnx_session_ = allocate_shared_session(policy_path);
    parse_g1_metadata(onnx_session_);
    policy_ = std::make_unique<OnnxPolicy>(onnx_session_);
}

py::tuple G1Rollout::rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& initial_state,
    const py::array_t<double>& controls
) {
    if (closed_) {
        throw std::runtime_error("Rollout requested after object was closed");
    }

    int B = (int)models.size();
    if (B == 0 || B != (int)data.size()) {
        throw std::runtime_error("models/data must have same non-zero length");
    }

    int horizon = (int)controls.shape(1);
    const mjModel* m0 = models[0];
    int nq = m0->nq;
    int nv = m0->nv;
    int nu = m0->nu;
    int nsens = m0->nsensordata;
    int nstate = nq + nv;

    if (initial_state.ndim() != 2 || initial_state.shape(0) != B || initial_state.shape(1) != nstate) {
        throw std::runtime_error("initial_state must be a 2D array of shape (B, nq+nv)");
    }

    // Controls should be (B, horizon, 17) for G1 - [vx, vy, wz, left_arm(7), right_arm(7)]
    // If controls are all zeros for arm indices, policy output will be used instead
    if (controls.ndim() != 3 || controls.shape(0) != B || controls.shape(2) != 17) {
        throw std::runtime_error("controls must be a 3D array of shape (B, horizon, 17)");
    }

    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sens_buf(B * horizon * nsens);

    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = initial_state.data();

    std::vector<std::vector<float>> prev_policy(B, std::vector<float>(g_g1_metadata.num_joints, 0.0f));

    {
        py::gil_scoped_release release;

        auto execute_work = [&](int i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            mjData* d = data[i];
            const mjModel* m = models[i];

            d->time = 0.0;
            const double* x0_i = x0_ptr + i * nstate;
            mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
            mj_forward(m, d);
            mju_zero(d->qacc_warmstart, m->nv);

            double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
            double* se_ptr = &sens_buf[i * horizon * nsens];

            // Store initial state
            for (int j = 0; j < nq; j++) st_ptr[j] = d->qpos[j];
            for (int j = 0; j < nv; j++) st_ptr[nq + j] = d->qvel[j];

            int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
            compute_indices(m, base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start);

            for (int t = 0; t < horizon; t++) {
                // Check timeout before each step
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                if (elapsed > cutoff_time_) {
                    // Timeout reached, fill remaining states with current state and return
                    for (int remaining_t = t; remaining_t < horizon; remaining_t++) {
                        for (int j = 0; j < nq; j++) st_ptr[(remaining_t + 1) * nstate + j] = d->qpos[j];
                        for (int j = 0; j < nv; j++) st_ptr[(remaining_t + 1) * nstate + nq + j] = d->qvel[j];
                        for (int j = 0; j < nsens; j++) se_ptr[remaining_t * nsens + j] = d->sensordata[j];
                    }
                    return;
                }

                // Get full control command [vx, vy, wz, left_arm(7), right_arm(7)]
                std::vector<float> obs;
                double cmd_vel_buf[3];
                double arm_cmd_buf[14];  // left_arm(7) + right_arm(7)

                // Extract velocity commands
                for (int j = 0; j < 3; j++) {
                    cmd_vel_buf[j] = static_cast<double>(controls_unchecked(i, t, j));
                }

                // Extract arm commands
                for (int j = 0; j < 14; j++) {
                    arm_cmd_buf[j] = static_cast<double>(controls_unchecked(i, t, 3 + j));
                }

                // Build observation for policy (using velocity commands only)
                build_observation(m, d, cmd_vel_buf, prev_policy[i],
                                base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start, obs);

                // Run policy
                auto policy_out_vec = policy_->run(obs);

                // Compute control from policy output and user arm commands
                std::vector<double> ctrl;
                compute_control_from_policy(policy_out_vec.data(), arm_cmd_buf, ctrl);

                if ((int)ctrl.size() != nu) {
                    throw std::runtime_error("Computed control size does not match model nu");
                }
                for (int j = 0; j < nu; j++) d->ctrl[j] = ctrl[j];

                // Step simulation
                mj_step(m, d);
                prev_policy[i] = std::move(policy_out_vec);

                // Store state and sensor data
                for (int j = 0; j < nq; j++) st_ptr[(t + 1) * nstate + j] = d->qpos[j];
                for (int j = 0; j < nv; j++) st_ptr[(t + 1) * nstate + nq + j] = d->qvel[j];
                for (int j = 0; j < nsens; j++) se_ptr[t * nsens + j] = d->sensordata[j];
            }
        };

        if (num_threads_ == 0) {
            // Single-threaded execution
            for (int i = 0; i < B; ++i) {
                execute_work(i);
            }
        } else {
            // Multi-threaded execution
            thread_pool_->execute_parallel(execute_work, B);
        }
    }

    auto states_arr = make_array_owned_g1(states_buf, B, horizon + 1, nstate);
    auto sens_arr = make_array_owned_g1(sens_buf, B, horizon, nsens);
    return py::make_tuple(states_arr, sens_arr);
}

G1Rollout* G1Rollout::__enter__() {
    return this;
}

void G1Rollout::__exit__(py::object exc_type, py::object exc_val, py::object exc_tb) {
    close();
}

int G1Rollout::get_num_threads() const {
    return num_threads_;
}

// Helper method implementations
void G1Rollout::compute_indices(const mjModel* m, int& base_qpos_start, int& base_qvel_start,
                                int& leg_qpos_start, int& leg_qvel_start) {
    // Find the free joint (root body)
    int free_joint_idx = -1;
    for (int j = 0; j < m->njnt; j++) {
        if (m->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = m->jnt_qposadr[free_joint_idx];
    base_qvel_start = m->jnt_dofadr[free_joint_idx];
    // G1 has 29 controlled joints starting after the free joint
    leg_qpos_start = base_qpos_start + 7;  // After pos(3) + quat(4)
    leg_qvel_start = base_qvel_start + 6;  // After lin_vel(3) + ang_vel(3)
}

void G1Rollout::build_observation(const mjModel* m, mjData* d, const double* command_ptr,
                                  const std::vector<float>& prev_policy,
                                  int base_qpos_start, int base_qvel_start,
                                  int leg_qpos_start, int leg_qvel_start,
                                  std::vector<float>& obs_out) {
    obs_out.resize(g_g1_metadata.obs_dim);
    int off = 0;

    // Get root body quaternion
    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = d->qpos[base_qpos_start + 3 + i];
    }

    // Compute inverse quaternion for body frame transformations
    double invq[4];
    mju_negQuat(invq, quat);

    // Compute rotation matrix from quaternion
    double rot_mat[9];
    mju_quat2Mat(rot_mat, quat);

    // Base linear velocity in body frame (3)
    double base_lin_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_lin_vel_world[i] = d->qvel[base_qvel_start + i];
    }
    double base_lin_vel_body[3];
    mju_rotVecQuat(base_lin_vel_body, base_lin_vel_world, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(base_lin_vel_body[i]);

    // Base angular velocity in body frame (3)
    double base_ang_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_ang_vel_world[i] = d->qvel[base_qvel_start + 3 + i];
    }
    double base_ang_vel_body[3];
    mju_rotVecQuat(base_ang_vel_body, base_ang_vel_world, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(base_ang_vel_body[i]);

    // Projected gravity in body frame (3)
    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs_out[off++] = static_cast<float>(gvec_rotated[i]);

    // Joint positions relative to default
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        double joint_pos = d->qpos[leg_qpos_start + i] - g_g1_metadata.default_joint_pos[i];
        obs_out[off++] = static_cast<float>(joint_pos);
    }

    // Joint velocities
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs_out[off++] = static_cast<float>(d->qvel[leg_qvel_start + i]);
    }

    // Previous actions
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs_out[off++] = (i < (int)prev_policy.size() ? prev_policy[i] : 0.0f);
    }

    // Command (3): [vx, vy, wz]
    for (int i = 0; i < 3; i++) {
        obs_out[off++] = static_cast<float>(command_ptr[i]);
    }
}

void G1Rollout::compute_control_from_policy(const float* policy_out,
                                           const double* arm_commands,
                                           std::vector<double>& ctrl_out) {
    ctrl_out.resize(g_g1_metadata.num_joints);

    // Apply action scaling and add default positions for all joints
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        ctrl_out[i] = static_cast<double>(policy_out[i]) * g_g1_metadata.action_scale[i] + g_g1_metadata.default_joint_pos[i];
    }

    // Override arm joints with user commands if provided (non-zero)
    // arm_commands layout: [left_arm(7), right_arm(7)]
    // arm_joint_indices contains the indices of arm joints in the full joint array
    for (size_t i = 0; i < g_g1_metadata.arm_joint_indices.size(); i++) {
        int joint_idx = g_g1_metadata.arm_joint_indices[i];
        double user_cmd = arm_commands[i];

        // If user command is non-zero, use it; otherwise use policy output
        if (std::abs(user_cmd) > 1e-8) {
            ctrl_out[joint_idx] = user_cmd;
        }
        // else: keep policy output (already set above)
    }
}

// =============================================================================
// SimG1 - Single-step simulation with G1 policy
// =============================================================================

py::array_t<float> SimG1(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& command,
    const py::array_t<float>& prev_policy
) {
    static std::shared_ptr<Ort::Session> onnx_session = nullptr;
    static std::unique_ptr<OnnxPolicy> policy = nullptr;

    // Initialize policy on first call
    if (!onnx_session || !policy) {
        std::string policy_path = get_g1_policy_path();
        onnx_session = allocate_shared_session(policy_path);
        parse_g1_metadata(onnx_session);
        policy = std::make_unique<OnnxPolicy>(onnx_session);
    }

    int nq = model->nq;
    int nv = model->nv;
    int nu = model->nu;

    if (x0.size() != nq + nv) {
        throw std::runtime_error("x0 size must equal nq + nv");
    }
    if (command.size() != 17) {
        throw std::runtime_error("command size must be 17 [vx, vy, wz, left_arm(7), right_arm(7)]");
    }
    if (prev_policy.size() != g_g1_metadata.num_joints) {
        throw std::runtime_error("prev_policy size must match num_joints from metadata");
    }

    // Set initial state
    const double* x0_ptr = x0.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);

    // Convert prev_policy to vector
    std::vector<float> prev_policy_vec(g_g1_metadata.num_joints);
    const float* prev_policy_ptr = prev_policy.data();
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        prev_policy_vec[i] = prev_policy_ptr[i];
    }

    // Get command data - extract velocity commands and arm commands
    const double* cmd_ptr = command.data();
    double cmd_vel[3] = {cmd_ptr[0], cmd_ptr[1], cmd_ptr[2]};
    double arm_cmd[14];  // left_arm(7) + right_arm(7)
    for (int i = 0; i < 14; i++) {
        arm_cmd[i] = cmd_ptr[3 + i];
    }

    // Compute indices for root and joint positions/velocities
    int base_qpos_start, base_qvel_start, leg_qpos_start, leg_qvel_start;
    int free_joint_idx = -1;
    for (int j = 0; j < model->njnt; j++) {
        if (model->jnt_type[j] == mjJNT_FREE) {
            free_joint_idx = j;
            break;
        }
    }
    if (free_joint_idx == -1) {
        free_joint_idx = 0;
    }
    base_qpos_start = model->jnt_qposadr[free_joint_idx];
    base_qvel_start = model->jnt_dofadr[free_joint_idx];
    leg_qpos_start = base_qpos_start + 7;
    leg_qvel_start = base_qvel_start + 6;

    // Build observation vector
    std::vector<float> obs(g_g1_metadata.obs_dim);
    int off = 0;

    // Get root body quaternion
    double quat[4];
    for (int i = 0; i < 4; i++) {
        quat[i] = data->qpos[base_qpos_start + 3 + i];
    }

    // Compute inverse quaternion for body frame transformations
    double invq[4];
    mju_negQuat(invq, quat);

    // Base linear velocity in body frame (3)
    double base_lin_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_lin_vel_world[i] = data->qvel[base_qvel_start + i];
    }
    double base_lin_vel_body[3];
    mju_rotVecQuat(base_lin_vel_body, base_lin_vel_world, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(base_lin_vel_body[i]);

    // Base angular velocity in body frame (3)
    double base_ang_vel_world[3];
    for (int i = 0; i < 3; i++) {
        base_ang_vel_world[i] = data->qvel[base_qvel_start + 3 + i];
    }
    double base_ang_vel_body[3];
    mju_rotVecQuat(base_ang_vel_body, base_ang_vel_world, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(base_ang_vel_body[i]);

    // Projected gravity in body frame (3)
    double gvec[3] = {0.0, 0.0, -1.0};
    double gvec_rotated[3];
    mju_rotVecQuat(gvec_rotated, gvec, invq);
    for (int i = 0; i < 3; i++) obs[off++] = static_cast<float>(gvec_rotated[i]);

    // Joint positions relative to default
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        double joint_pos = data->qpos[leg_qpos_start + i] - g_g1_metadata.default_joint_pos[i];
        obs[off++] = static_cast<float>(joint_pos);
    }

    // Joint velocities
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs[off++] = static_cast<float>(data->qvel[leg_qvel_start + i]);
    }

    // Previous actions
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        obs[off++] = prev_policy_vec[i];
    }

    // Command (3): [vx, vy, wz] - use velocity commands only
    for (int i = 0; i < 3; i++) {
        obs[off++] = static_cast<float>(cmd_vel[i]);
    }

    // Run policy
    auto policy_out_vec = policy->run(obs);

    // Compute control from policy output
    std::vector<double> ctrl(g_g1_metadata.num_joints);

    // Apply action scaling and add default positions for all joints
    for (int i = 0; i < g_g1_metadata.num_joints; i++) {
        ctrl[i] = static_cast<double>(policy_out_vec[i]) * g_g1_metadata.action_scale[i] + g_g1_metadata.default_joint_pos[i];
    }

    // Override arm joints with user commands if provided (non-zero)
    // arm_cmd layout: [left_arm(7), right_arm(7)]
    for (size_t i = 0; i < g_g1_metadata.arm_joint_indices.size(); i++) {
        int joint_idx = g_g1_metadata.arm_joint_indices[i];
        double user_cmd = arm_cmd[i];

        // If user command is non-zero, use it; otherwise use policy output
        if (std::abs(user_cmd) > 1e-8) {
            ctrl[joint_idx] = user_cmd;
        }
        // else: keep policy output (already set above)
    }

    // Apply control and step simulation
    if (nu != g_g1_metadata.num_joints) {
        throw std::runtime_error("Model nu does not match num_joints from metadata");
    }
    for (int j = 0; j < nu; j++) {
        data->ctrl[j] = ctrl[j];
    }
    mj_step(model, data);

    // Return new policy output as numpy array
    std::vector<ssize_t> shape = {g_g1_metadata.num_joints};
    std::vector<ssize_t> strides = {sizeof(float)};
    auto heap_buf = new std::vector<float>(std::move(policy_out_vec));
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<float>*>(p); });
    return py::array_t<float>(shape, strides, heap_buf->data(), free_when_done);
}

// =============================================================================
// G1 WBC native rollout/simulation
// =============================================================================

namespace {

constexpr int WBC_ACTION_DIM = 29;
constexpr int WBC_CONTROL_DIM = 36;
constexpr int WBC_OBS_DIM = 886;
constexpr int WBC_POLICY_DECIMATION = 4;
constexpr int WBC_HISTORY_LEN = 5;
constexpr int WBC_LIMB_COUNT = 4;
constexpr int WBC_BODY_COUNT = 30;
constexpr int WBC_LIMB_POSE_DIM = 36;
constexpr int WBC_CONTACT_SENSOR_DIM = 4;
constexpr int WBC_UPPER_EE_SENSOR_DIM = 6;
constexpr int WBC_ACTION_SENSOR_DIM = WBC_ACTION_DIM;
constexpr int WBC_SENSOR_DIM = WBC_CONTACT_SENSOR_DIM + WBC_UPPER_EE_SENSOR_DIM + WBC_ACTION_SENSOR_DIM;
constexpr int WBC_POLICY_STATE_DIM =
    2 + WBC_CONTROL_DIM + WBC_ACTION_DIM + WBC_ACTION_DIM + WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM +
    WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM + WBC_HISTORY_LEN * 3 + WBC_HISTORY_LEN * 3 +
    WBC_HISTORY_LEN * WBC_ACTION_DIM + WBC_HISTORY_LEN * WBC_ACTION_DIM + WBC_HISTORY_LEN * WBC_ACTION_DIM;

const std::array<double, WBC_ACTION_DIM> WBC_DEFAULT_JOINT_POS = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0,
    0.2,    0.2, 0.0, 0.6,   0.0,    0.0, 0.0,    0.2, -0.2, 0.0, 0.6,   0.0,    0.0, 0.0,
};

const std::array<double, WBC_ACTION_DIM> WBC_ACTION_SCALE = {
    0.54754646299110676, 0.35066146637882434, 0.54754646299110676, 0.35066146637882434,
    0.43857731392336724, 0.43857731392336724, 0.54754646299110676, 0.35066146637882434,
    0.54754646299110676, 0.35066146637882434, 0.43857731392336724, 0.43857731392336724,
    0.54754646299110676, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.074500870329507141, 0.074500870329507141, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724, 0.074500870329507141,
    0.074500870329507141,
};

const std::array<const char*, WBC_BODY_COUNT> WBC_BODY_NAMES = {
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
};

constexpr std::array<int, WBC_LIMB_COUNT> WBC_LIMB_BODY_INDICES = {22, 29, 6, 12};
constexpr std::array<int, 2> WBC_UPPER_EE_BODY_INDICES = {22, 29};
constexpr int WBC_ANCHOR_BODY_INDEX = 0;
constexpr int WBC_POLICY_ANG_VEL_BODY_INDEX = 15;

struct WBCBodyIds {
    std::array<int, WBC_BODY_COUNT> body_ids{};
};

struct WBCPolicyState {
    int step_count = 0;
    bool initialized = false;
    std::array<float, WBC_CONTROL_DIM> last_control{};
    std::array<float, WBC_ACTION_DIM> last_action{};
    std::array<float, WBC_ACTION_DIM> held_ctrl{};
    std::array<float, WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM> ref_limb_hist{};
    std::array<float, WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM> robot_limb_hist{};
    std::array<float, WBC_HISTORY_LEN * 3> gravity_hist{};
    std::array<float, WBC_HISTORY_LEN * 3> base_ang_vel_hist{};
    std::array<float, WBC_HISTORY_LEN * WBC_ACTION_DIM> joint_pos_hist{};
    std::array<float, WBC_HISTORY_LEN * WBC_ACTION_DIM> joint_vel_hist{};
    std::array<float, WBC_HISTORY_LEN * WBC_ACTION_DIM> action_hist{};

    void reset() {
        step_count = 0;
        initialized = false;
        last_control.fill(0.0f);
        last_action.fill(0.0f);
        for (int i = 0; i < WBC_ACTION_DIM; ++i) {
            held_ctrl[i] = static_cast<float>(WBC_DEFAULT_JOINT_POS[i]);
        }
        ref_limb_hist.fill(0.0f);
        robot_limb_hist.fill(0.0f);
        gravity_hist.fill(0.0f);
        base_ang_vel_hist.fill(0.0f);
        joint_pos_hist.fill(0.0f);
        joint_vel_hist.fill(0.0f);
        action_hist.fill(0.0f);
    }
};

static WBCBodyIds resolve_wbc_body_ids(const mjModel* model) {
    WBCBodyIds ids;
    for (int i = 0; i < static_cast<int>(WBC_BODY_NAMES.size()); ++i) {
        int body_id = mj_name2id(model, mjOBJ_BODY, WBC_BODY_NAMES[i]);
        if (body_id < 0) {
            throw std::runtime_error(std::string("Missing G1 WBC body in MuJoCo model: ") + WBC_BODY_NAMES[i]);
        }
        ids.body_ids[i] = body_id;
    }
    return ids;
}

static std::string default_wbc_policy_path() {
    const char* explicit_path = std::getenv("SUMO_G1_WBC_POLICY_PATH");
    if (explicit_path && explicit_path[0] != '\0') {
        return std::string(explicit_path);
    }
    const char* repo_root = std::getenv("SUMO_REPO_ROOT");
    if (repo_root && repo_root[0] != '\0') {
        const char* variant = std::getenv("SUMO_G1_WBC_POLICY");
        std::string policy = variant ? std::string(variant) : std::string("bcrl");
        if (policy == "bc") {
            return std::string(repo_root) + "/../wxy/0608_ckpt_bc/deploy_model_8000.onnx";
        }
        return std::string(repo_root) + "/../wxy/0608_ckpt_bcrl/deploy_model_16000.onnx";
    }
    return std::string("../wxy/0608_ckpt_bcrl/deploy_model_16000.onnx");
}

static OnnxPolicy* cached_single_step_wbc_policy(const std::string& policy_path) {
    static std::shared_ptr<Ort::Session> onnx_session = nullptr;
    static std::unique_ptr<OnnxPolicy> policy = nullptr;
    static std::string loaded_policy_path;

    std::string resolved_policy_path = policy_path.empty() ? default_wbc_policy_path() : policy_path;
    if (!onnx_session || !policy || loaded_policy_path != resolved_policy_path) {
        onnx_session = allocate_shared_session(resolved_policy_path);
        policy = std::make_unique<OnnxPolicy>(onnx_session);
        if (policy->input_size() != WBC_OBS_DIM || policy->output_size() != WBC_ACTION_DIM) {
            throw std::runtime_error("G1 WBC ONNX policy must have input 886 and output 29");
        }
        loaded_policy_path = resolved_policy_path;
    }
    return policy.get();
}

static void normalize_quat_inplace(double q[4]) {
    double norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norm <= 1e-12) {
        q[0] = 1.0;
        q[1] = q[2] = q[3] = 0.0;
        return;
    }
    for (int i = 0; i < 4; ++i) {
        q[i] /= norm;
    }
    if (q[0] < 0.0) {
        for (int i = 0; i < 4; ++i) {
            q[i] = -q[i];
        }
    }
}

static void control_to_qpos(const double* control, double* qpos_out) {
    for (int i = 0; i < 3; ++i) {
        qpos_out[i] = control[i];
    }
    double quat[4] = {control[3], control[4], control[5], control[6]};
    normalize_quat_inplace(quat);
    for (int i = 0; i < 4; ++i) {
        qpos_out[3 + i] = quat[i];
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        qpos_out[7 + i] = control[7 + i];
    }
}

static void quat_to_rotvec(const double* quat_in, double* rotvec_out) {
    double q[4] = {quat_in[0], quat_in[1], quat_in[2], quat_in[3]};
    normalize_quat_inplace(q);
    double w = std::max(-1.0, std::min(1.0, q[0]));
    double angle = 2.0 * std::acos(w);
    double s = std::sqrt(std::max(0.0, 1.0 - w * w));
    if (s < 1e-8) {
        rotvec_out[0] = 2.0 * q[1];
        rotvec_out[1] = 2.0 * q[2];
        rotvec_out[2] = 2.0 * q[3];
        return;
    }
    rotvec_out[0] = angle * q[1] / s;
    rotvec_out[1] = angle * q[2] / s;
    rotvec_out[2] = angle * q[3] / s;
}

static void qvel_from_control_pair(const double* previous, const double* current, double dt, double* qvel_out) {
    std::fill(qvel_out, qvel_out + 35, 0.0);
    if (previous == nullptr || dt <= 0.0) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        qvel_out[i] = (current[i] - previous[i]) / dt;
    }
    double prev_rv[3];
    double curr_rv[3];
    quat_to_rotvec(previous + 3, prev_rv);
    quat_to_rotvec(current + 3, curr_rv);
    for (int i = 0; i < 3; ++i) {
        qvel_out[3 + i] = (curr_rv[i] - prev_rv[i]) / dt;
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        qvel_out[6 + i] = (current[7 + i] - previous[7 + i]) / dt;
    }
}

static void qvel_from_control_sequence(
    const py::detail::unchecked_reference<double, 3>& controls,
    int batch,
    int t,
    int horizon,
    double dt,
    double* qvel_out
) {
    std::fill(qvel_out, qvel_out + 35, 0.0);
    if (horizon <= 1 || dt <= 0.0) {
        return;
    }
    int lhs = t == 0 ? 0 : t - 1;
    int rhs = t == horizon - 1 ? horizon - 1 : t + 1;
    double scale = 1.0 / ((rhs - lhs) * dt);
    if (rhs == lhs) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        qvel_out[i] = (controls(batch, rhs, i) - controls(batch, lhs, i)) * scale;
    }
    double lhs_q[4] = {
        controls(batch, lhs, 3),
        controls(batch, lhs, 4),
        controls(batch, lhs, 5),
        controls(batch, lhs, 6),
    };
    double rhs_q[4] = {
        controls(batch, rhs, 3),
        controls(batch, rhs, 4),
        controls(batch, rhs, 5),
        controls(batch, rhs, 6),
    };
    double lhs_rv[3];
    double rhs_rv[3];
    quat_to_rotvec(lhs_q, lhs_rv);
    quat_to_rotvec(rhs_q, rhs_rv);
    for (int i = 0; i < 3; ++i) {
        qvel_out[3 + i] = (rhs_rv[i] - lhs_rv[i]) * scale;
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        qvel_out[6 + i] = (controls(batch, rhs, 7 + i) - controls(batch, lhs, 7 + i)) * scale;
    }
}

static void overwrite_policy_ang_vel_from_control_pair(
    const mjModel* model,
    mjData* ref_data,
    const WBCBodyIds& ids,
    const double* previous,
    const double* current,
    double dt,
    double* qvel_out
) {
    if (previous == nullptr || dt <= 0.0) {
        return;
    }

    double prev_rotvec[3];
    double curr_rotvec[3];
    int body_id = ids.body_ids[WBC_POLICY_ANG_VEL_BODY_INDEX];

    control_to_qpos(previous, ref_data->qpos);
    mju_zero(ref_data->qvel, model->nv);
    mj_forward(model, ref_data);
    quat_to_rotvec(ref_data->xquat + 4 * body_id, prev_rotvec);

    control_to_qpos(current, ref_data->qpos);
    mju_zero(ref_data->qvel, model->nv);
    mj_forward(model, ref_data);
    quat_to_rotvec(ref_data->xquat + 4 * body_id, curr_rotvec);

    for (int i = 0; i < 3; ++i) {
        qvel_out[3 + i] = (curr_rotvec[i] - prev_rotvec[i]) / dt;
    }
}

static void qvel_from_control_sequence_policy(
    const mjModel* model,
    mjData* ref_data,
    const WBCBodyIds& ids,
    const py::detail::unchecked_reference<double, 3>& controls,
    int batch,
    int horizon,
    double dt,
    std::vector<double>& qvel_buf
) {
    qvel_buf.assign(static_cast<size_t>(horizon) * 35, 0.0);
    if (horizon <= 0) {
        return;
    }

    std::vector<double> torso_rotvec(static_cast<size_t>(horizon) * 3, 0.0);
    std::array<double, WBC_CONTROL_DIM> control{};
    int body_id = ids.body_ids[WBC_POLICY_ANG_VEL_BODY_INDEX];
    for (int t = 0; t < horizon; ++t) {
        double* qvel = qvel_buf.data() + static_cast<size_t>(t) * 35;
        qvel_from_control_sequence(controls, batch, t, horizon, dt, qvel);
        for (int j = 0; j < WBC_CONTROL_DIM; ++j) {
            control[j] = controls(batch, t, j);
        }
        control_to_qpos(control.data(), ref_data->qpos);
        for (int j = 0; j < model->nv; ++j) {
            ref_data->qvel[j] = qvel[j];
        }
        mj_forward(model, ref_data);
        quat_to_rotvec(ref_data->xquat + 4 * body_id, torso_rotvec.data() + static_cast<size_t>(t) * 3);
    }

    if (horizon <= 1 || dt <= 0.0) {
        for (int t = 0; t < horizon; ++t) {
            double* qvel = qvel_buf.data() + static_cast<size_t>(t) * 35;
            qvel[3] = qvel[4] = qvel[5] = 0.0;
        }
        return;
    }

    for (int t = 0; t < horizon; ++t) {
        int lhs = t == 0 ? 0 : t - 1;
        int rhs = t == horizon - 1 ? horizon - 1 : t + 1;
        double scale = 1.0 / ((rhs - lhs) * dt);
        double* qvel = qvel_buf.data() + static_cast<size_t>(t) * 35;
        const double* lhs_rv = torso_rotvec.data() + static_cast<size_t>(lhs) * 3;
        const double* rhs_rv = torso_rotvec.data() + static_cast<size_t>(rhs) * 3;
        for (int j = 0; j < 3; ++j) {
            qvel[3 + j] = (rhs_rv[j] - lhs_rv[j]) * scale;
        }
    }
}

static bool qvel_from_optional_array(const py::object& reference_qvel, int nv, double* qvel_out) {
    if (reference_qvel.is_none()) {
        return false;
    }
    py::array_t<double, py::array::c_style | py::array::forcecast> qvel_arr(reference_qvel);
    if (qvel_arr.ndim() != 1 || qvel_arr.shape(0) != nv) {
        throw std::runtime_error("reference_qvel must be None or a 1D array of shape (nv,)");
    }
    std::memcpy(qvel_out, qvel_arr.data(), sizeof(double) * nv);
    return true;
}

template <size_t N>
static void append_history(std::array<float, N>& hist, const float* value, int dim, bool initialized) {
    if (!initialized) {
        for (int h = 0; h < WBC_HISTORY_LEN; ++h) {
            std::memcpy(hist.data() + h * dim, value, sizeof(float) * dim);
        }
        return;
    }
    std::memmove(hist.data(), hist.data() + dim, sizeof(float) * dim * (WBC_HISTORY_LEN - 1));
    std::memcpy(hist.data() + dim * (WBC_HISTORY_LEN - 1), value, sizeof(float) * dim);
}

static void copy_array_to_obs(const float* ptr, int count, std::vector<float>& obs, int& offset) {
    std::memcpy(obs.data() + offset, ptr, sizeof(float) * count);
    offset += count;
}

static void limb_pose_in_anchor_frame(
    const mjData* data,
    const WBCBodyIds& ids,
    std::array<float, WBC_LIMB_POSE_DIM>& out
) {
    const int anchor_body_id = ids.body_ids[WBC_ANCHOR_BODY_INDEX];
    const double* anchor_pos = data->xpos + 3 * anchor_body_id;
    const double* anchor_quat = data->xquat + 4 * anchor_body_id;
    double inv_anchor_quat[4];
    mju_negQuat(inv_anchor_quat, anchor_quat);

    int off = 0;
    for (int limb_idx : WBC_LIMB_BODY_INDICES) {
        int body_id = ids.body_ids[limb_idx];
        const double* body_pos = data->xpos + 3 * body_id;
        const double* body_quat = data->xquat + 4 * body_id;

        double pos_delta[3] = {
            body_pos[0] - anchor_pos[0],
            body_pos[1] - anchor_pos[1],
            body_pos[2] - anchor_pos[2],
        };
        double pos_b[3];
        mju_rotVecQuat(pos_b, pos_delta, inv_anchor_quat);
        for (double v : pos_b) {
            out[off++] = static_cast<float>(v);
        }

        double quat_b[4];
        mju_mulQuat(quat_b, inv_anchor_quat, body_quat);
        normalize_quat_inplace(quat_b);
        double mat[9];
        mju_quat2Mat(mat, quat_b);
        out[off++] = static_cast<float>(mat[0]);
        out[off++] = static_cast<float>(mat[1]);
        out[off++] = static_cast<float>(mat[3]);
        out[off++] = static_cast<float>(mat[4]);
        out[off++] = static_cast<float>(mat[6]);
        out[off++] = static_cast<float>(mat[7]);
    }
}

static void build_wbc_observation(
    const mjModel* model,
    mjData* data,
    mjData* ref_data,
    const WBCBodyIds& ids,
    const double* control,
    const double* ref_qvel,
    WBCPolicyState& state,
    std::vector<float>& obs
) {
    obs.assign(WBC_OBS_DIM, 0.0f);

    std::array<float, WBC_ACTION_DIM> ref_joint_pos{};
    std::array<float, WBC_ACTION_DIM> ref_joint_vel{};
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        ref_joint_pos[i] = static_cast<float>(control[7 + i]);
        ref_joint_vel[i] = static_cast<float>(ref_qvel[6 + i]);
    }

    control_to_qpos(control, ref_data->qpos);
    for (int i = 0; i < model->nv; ++i) {
        ref_data->qvel[i] = ref_qvel[i];
    }
    mj_forward(model, ref_data);

    std::array<float, WBC_LIMB_POSE_DIM> ref_limb{};
    std::array<float, WBC_LIMB_POSE_DIM> robot_limb{};
    limb_pose_in_anchor_frame(ref_data, ids, ref_limb);
    limb_pose_in_anchor_frame(data, ids, robot_limb);

    double root_quat[4] = {data->qpos[3], data->qpos[4], data->qpos[5], data->qpos[6]};
    normalize_quat_inplace(root_quat);
    double inv_root_quat[4];
    mju_negQuat(inv_root_quat, root_quat);

    double gravity_w[3] = {0.0, 0.0, -1.0};
    double gravity_b[3];
    mju_rotVecQuat(gravity_b, gravity_w, inv_root_quat);

    std::array<float, 3> gravity{};
    std::array<float, 3> base_ang_vel{};
    std::array<float, WBC_ACTION_DIM> joint_pos_rel{};
    std::array<float, WBC_ACTION_DIM> joint_vel{};
    for (int i = 0; i < 3; ++i) {
        gravity[i] = static_cast<float>(gravity_b[i]);
        base_ang_vel[i] = static_cast<float>(data->qvel[3 + i]);
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        joint_pos_rel[i] = static_cast<float>(data->qpos[7 + i] - WBC_DEFAULT_JOINT_POS[i]);
        joint_vel[i] = static_cast<float>(data->qvel[6 + i]);
    }

    bool initialized = state.initialized;
    append_history(state.ref_limb_hist, ref_limb.data(), WBC_LIMB_POSE_DIM, initialized);
    append_history(state.robot_limb_hist, robot_limb.data(), WBC_LIMB_POSE_DIM, initialized);
    append_history(state.gravity_hist, gravity.data(), 3, initialized);
    append_history(state.base_ang_vel_hist, base_ang_vel.data(), 3, initialized);
    append_history(state.joint_pos_hist, joint_pos_rel.data(), WBC_ACTION_DIM, initialized);
    append_history(state.joint_vel_hist, joint_vel.data(), WBC_ACTION_DIM, initialized);
    append_history(state.action_hist, state.last_action.data(), WBC_ACTION_DIM, initialized);
    state.initialized = true;

    int off = 0;
    copy_array_to_obs(ref_joint_pos.data(), WBC_ACTION_DIM, obs, off);
    copy_array_to_obs(ref_joint_vel.data(), WBC_ACTION_DIM, obs, off);
    copy_array_to_obs(state.ref_limb_hist.data(), WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM, obs, off);
    for (int i = 0; i < 3; ++i) {
        obs[off++] = static_cast<float>(ref_qvel[3 + i]);
    }
    copy_array_to_obs(state.robot_limb_hist.data(), WBC_HISTORY_LEN * WBC_LIMB_POSE_DIM, obs, off);
    copy_array_to_obs(state.gravity_hist.data(), WBC_HISTORY_LEN * 3, obs, off);
    copy_array_to_obs(state.base_ang_vel_hist.data(), WBC_HISTORY_LEN * 3, obs, off);
    copy_array_to_obs(state.joint_pos_hist.data(), WBC_HISTORY_LEN * WBC_ACTION_DIM, obs, off);
    copy_array_to_obs(state.joint_vel_hist.data(), WBC_HISTORY_LEN * WBC_ACTION_DIM, obs, off);
    copy_array_to_obs(state.action_hist.data(), WBC_HISTORY_LEN * WBC_ACTION_DIM, obs, off);

    if (off != WBC_OBS_DIM) {
        throw std::runtime_error("Internal G1 WBC observation dimension mismatch");
    }
}

static void apply_wbc_policy_step(
    const mjModel* model,
    mjData* data,
    mjData* ref_data,
    const WBCBodyIds& ids,
    OnnxPolicy* policy,
    const double* control,
    const double* ref_qvel,
    WBCPolicyState& state
) {
    if (state.step_count % WBC_POLICY_DECIMATION == 0) {
        std::vector<float> obs;
        build_wbc_observation(model, data, ref_data, ids, control, ref_qvel, state, obs);
        std::vector<float> policy_out = policy->run(obs);
        if (static_cast<int>(policy_out.size()) != WBC_ACTION_DIM) {
            throw std::runtime_error("G1 WBC policy output dimension must be 29");
        }
        for (int i = 0; i < WBC_ACTION_DIM; ++i) {
            state.last_action[i] = policy_out[i];
            state.held_ctrl[i] = static_cast<float>(policy_out[i] * WBC_ACTION_SCALE[i] + WBC_DEFAULT_JOINT_POS[i]);
        }
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        data->ctrl[i] = static_cast<double>(state.held_ctrl[i]);
    }
    state.step_count += 1;
}

static void contact_sensor_values_cpp(const mjModel* model, mjData* data, double* out) {
    out[0] = out[1] = out[2] = out[3] = 0.0;
    double force6[6];
    for (int contact_idx = 0; contact_idx < data->ncon; ++contact_idx) {
        const mjContact& contact = data->contact[contact_idx];
        int sides[2] = {-1, -1};
        int side_count = 0;
        const char* geom1 = mj_id2name(model, mjOBJ_GEOM, contact.geom1);
        const char* geom2 = mj_id2name(model, mjOBJ_GEOM, contact.geom2);
        if (geom1 && std::strncmp(geom1, "left_foot", 9) == 0) {
            sides[side_count++] = 0;
        } else if (geom1 && std::strncmp(geom1, "right_foot", 10) == 0) {
            sides[side_count++] = 1;
        }
        if (geom2 && std::strncmp(geom2, "left_foot", 9) == 0) {
            sides[side_count++] = 0;
        } else if (geom2 && std::strncmp(geom2, "right_foot", 10) == 0) {
            sides[side_count++] = 1;
        }
        if (side_count == 0) {
            continue;
        }
        mj_contactForce(model, data, contact_idx, force6);
        double force_mag = std::sqrt(force6[0] * force6[0] + force6[1] * force6[1] + force6[2] * force6[2]);
        for (int i = 0; i < side_count; ++i) {
            int side = sides[i];
            out[side] = 1.0;
            out[2 + side] += force_mag;
        }
    }
}

static void wbc_sensor_values_cpp(
    const mjModel* model,
    mjData* data,
    const WBCBodyIds& ids,
    const WBCPolicyState& state,
    double* out
) {
    std::fill(out, out + WBC_SENSOR_DIM, 0.0);
    contact_sensor_values_cpp(model, data, out);
    int off = WBC_CONTACT_SENSOR_DIM;
    for (int body_idx : WBC_UPPER_EE_BODY_INDICES) {
        const int body_id = ids.body_ids[body_idx];
        const double* pos = data->xpos + 3 * body_id;
        out[off++] = pos[0];
        out[off++] = pos[1];
        out[off++] = pos[2];
    }
    for (int i = 0; i < WBC_ACTION_DIM; ++i) {
        out[off++] = static_cast<double>(state.held_ctrl[i]);
    }
}

static void state_from_array(const py::array_t<float>& array, WBCPolicyState& state) {
    state.reset();
    if (array.ndim() != 1 || array.shape(0) != WBC_POLICY_STATE_DIM) {
        throw std::runtime_error("prev_policy_state must be a 1D array with g1_wbc_policy_state_dim() elements");
    }
    const float* ptr = array.data();
    state.step_count = static_cast<int>(ptr[0]);
    state.initialized = ptr[1] > 0.5f;
    int off = 2;
    std::memcpy(state.last_control.data(), ptr + off, sizeof(float) * WBC_CONTROL_DIM);
    off += WBC_CONTROL_DIM;
    std::memcpy(state.last_action.data(), ptr + off, sizeof(float) * WBC_ACTION_DIM);
    off += WBC_ACTION_DIM;
    std::memcpy(state.held_ctrl.data(), ptr + off, sizeof(float) * WBC_ACTION_DIM);
    off += WBC_ACTION_DIM;
    std::memcpy(state.ref_limb_hist.data(), ptr + off, sizeof(float) * state.ref_limb_hist.size());
    off += static_cast<int>(state.ref_limb_hist.size());
    std::memcpy(state.robot_limb_hist.data(), ptr + off, sizeof(float) * state.robot_limb_hist.size());
    off += static_cast<int>(state.robot_limb_hist.size());
    std::memcpy(state.gravity_hist.data(), ptr + off, sizeof(float) * state.gravity_hist.size());
    off += static_cast<int>(state.gravity_hist.size());
    std::memcpy(state.base_ang_vel_hist.data(), ptr + off, sizeof(float) * state.base_ang_vel_hist.size());
    off += static_cast<int>(state.base_ang_vel_hist.size());
    std::memcpy(state.joint_pos_hist.data(), ptr + off, sizeof(float) * state.joint_pos_hist.size());
    off += static_cast<int>(state.joint_pos_hist.size());
    std::memcpy(state.joint_vel_hist.data(), ptr + off, sizeof(float) * state.joint_vel_hist.size());
    off += static_cast<int>(state.joint_vel_hist.size());
    std::memcpy(state.action_hist.data(), ptr + off, sizeof(float) * state.action_hist.size());
}

static py::array_t<float> state_to_array(const WBCPolicyState& state) {
    auto heap_buf = new std::vector<float>(WBC_POLICY_STATE_DIM, 0.0f);
    std::vector<float>& buf = *heap_buf;
    buf[0] = static_cast<float>(state.step_count);
    buf[1] = state.initialized ? 1.0f : 0.0f;
    int off = 2;
    std::memcpy(buf.data() + off, state.last_control.data(), sizeof(float) * WBC_CONTROL_DIM);
    off += WBC_CONTROL_DIM;
    std::memcpy(buf.data() + off, state.last_action.data(), sizeof(float) * WBC_ACTION_DIM);
    off += WBC_ACTION_DIM;
    std::memcpy(buf.data() + off, state.held_ctrl.data(), sizeof(float) * WBC_ACTION_DIM);
    off += WBC_ACTION_DIM;
    std::memcpy(buf.data() + off, state.ref_limb_hist.data(), sizeof(float) * state.ref_limb_hist.size());
    off += static_cast<int>(state.ref_limb_hist.size());
    std::memcpy(buf.data() + off, state.robot_limb_hist.data(), sizeof(float) * state.robot_limb_hist.size());
    off += static_cast<int>(state.robot_limb_hist.size());
    std::memcpy(buf.data() + off, state.gravity_hist.data(), sizeof(float) * state.gravity_hist.size());
    off += static_cast<int>(state.gravity_hist.size());
    std::memcpy(buf.data() + off, state.base_ang_vel_hist.data(), sizeof(float) * state.base_ang_vel_hist.size());
    off += static_cast<int>(state.base_ang_vel_hist.size());
    std::memcpy(buf.data() + off, state.joint_pos_hist.data(), sizeof(float) * state.joint_pos_hist.size());
    off += static_cast<int>(state.joint_pos_hist.size());
    std::memcpy(buf.data() + off, state.joint_vel_hist.data(), sizeof(float) * state.joint_vel_hist.size());
    off += static_cast<int>(state.joint_vel_hist.size());
    std::memcpy(buf.data() + off, state.action_hist.data(), sizeof(float) * state.action_hist.size());

    std::vector<ssize_t> shape = {WBC_POLICY_STATE_DIM};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(sizeof(float))};
    py::capsule free_when_done(heap_buf, [](void *p) { delete reinterpret_cast<std::vector<float>*>(p); });
    return py::array_t<float>(shape, strides, heap_buf->data(), free_when_done);
}

static py::array_t<float> copy_to_float_array(const float* src, int count) {
    py::array_t<float> out(count);
    std::memcpy(out.mutable_data(), src, sizeof(float) * count);
    return out;
}

}  // namespace

int G1WBCPolicyStateDim() {
    return WBC_POLICY_STATE_DIM;
}

G1WBCRollout::G1WBCRollout(int nthread, double cutoff_time, const std::string& policy_path)
    : num_threads_(nthread), cutoff_time_(cutoff_time), policy_path_(policy_path.empty() ? default_wbc_policy_path() : policy_path) {
    initialize_policy();
    if (num_threads_ != 0) {
        thread_pool_ = std::make_unique<G1ThreadPool>(num_threads_);
    }
}

G1WBCRollout::~G1WBCRollout() {
    close();
}

void G1WBCRollout::close() {
    if (!closed_) {
        thread_pool_.reset();
        policy_.reset();
        onnx_session_.reset();
        closed_ = true;
    }
}

void G1WBCRollout::initialize_policy() {
    onnx_session_ = allocate_shared_session(policy_path_);
    policy_ = std::make_unique<OnnxPolicy>(onnx_session_);
    if (policy_->input_size() != WBC_OBS_DIM || policy_->output_size() != WBC_ACTION_DIM) {
        throw std::runtime_error("G1 WBC ONNX policy must have input 886 and output 29");
    }
}

py::tuple G1WBCRollout::rollout(
    const std::vector<const mjModel*>& models,
    const std::vector<mjData*>& data,
    const py::array_t<double>& initial_state,
    const py::array_t<double>& controls,
    const py::array_t<float>& initial_policy_state,
    const py::object& reference_qvels
) {
    if (closed_) {
        throw std::runtime_error("Rollout requested after object was closed");
    }

    int B = static_cast<int>(models.size());
    if (B == 0 || B != static_cast<int>(data.size())) {
        throw std::runtime_error("models/data must have same non-zero length");
    }
    if (controls.ndim() != 3 || controls.shape(0) != B || controls.shape(2) != WBC_CONTROL_DIM) {
        throw std::runtime_error("controls must be a 3D array of shape (B, horizon, 36)");
    }

    int horizon = static_cast<int>(controls.shape(1));
    const mjModel* m0 = models[0];
    int nq = m0->nq;
    int nv = m0->nv;
    int nstate = nq + nv;

    if (nq != WBC_CONTROL_DIM || nv != 35 || m0->nu != WBC_ACTION_DIM) {
        throw std::runtime_error("G1 WBC native backend expects nq=36, nv=35, nu=29");
    }
    if (initial_state.ndim() != 2 || initial_state.shape(0) != B || initial_state.shape(1) != nstate) {
        throw std::runtime_error("initial_state must be a 2D array of shape (B, nq+nv)");
    }

    std::unique_ptr<py::array_t<double, py::array::c_style | py::array::forcecast>> reference_qvel_arr;
    const double* reference_qvel_ptr = nullptr;
    if (!reference_qvels.is_none()) {
        reference_qvel_arr = std::make_unique<py::array_t<double, py::array::c_style | py::array::forcecast>>(reference_qvels);
        if (reference_qvel_arr->ndim() != 3 ||
            reference_qvel_arr->shape(0) != B ||
            reference_qvel_arr->shape(1) != horizon ||
            reference_qvel_arr->shape(2) != nv) {
            throw std::runtime_error("reference_qvels must be None or an array of shape (B, horizon, nv)");
        }
        reference_qvel_ptr = reference_qvel_arr->data();
    }

    WBCPolicyState base_policy_state;
    state_from_array(initial_policy_state, base_policy_state);

    std::vector<double> states_buf(B * (horizon + 1) * nstate);
    std::vector<double> sensor_buf(B * horizon * WBC_SENSOR_DIM);
    auto controls_unchecked = controls.unchecked<3>();
    const double* x0_ptr = initial_state.data();

    {
        py::gil_scoped_release release;

        auto execute_work = [&](int i) {
            auto start_time = std::chrono::high_resolution_clock::now();
            const mjModel* m = models[i];
            mjData* d = data[i];
            mjData* ref_data = mj_makeData(m);
            if (ref_data == nullptr) {
                throw std::runtime_error("Failed to allocate MuJoCo reference data for G1 WBC rollout");
            }

            WBCPolicyState policy_state = base_policy_state;
            WBCBodyIds ids = resolve_wbc_body_ids(m);
            std::vector<double> reference_qvel;
            const double* external_qvel = nullptr;
            if (reference_qvel_ptr != nullptr) {
                external_qvel = reference_qvel_ptr + static_cast<size_t>(i) * horizon * nv;
            }
            if (external_qvel != nullptr && std::isfinite(external_qvel[0])) {
                reference_qvel.assign(external_qvel, external_qvel + static_cast<size_t>(horizon) * nv);
            } else {
                qvel_from_control_sequence_policy(
                    m,
                    ref_data,
                    ids,
                    controls_unchecked,
                    i,
                    horizon,
                    m->opt.timestep,
                    reference_qvel
                );
            }

            d->time = 0.0;
            const double* x0_i = x0_ptr + i * nstate;
            mj_setState(m, d, x0_i, mjSTATE_QPOS | mjSTATE_QVEL);
            mj_forward(m, d);
            mju_zero(d->qacc_warmstart, m->nv);

            double* st_ptr = &states_buf[i * (horizon + 1) * nstate];
            double* se_ptr = &sensor_buf[i * horizon * WBC_SENSOR_DIM];
            for (int j = 0; j < nq; ++j) {
                st_ptr[j] = d->qpos[j];
            }
            for (int j = 0; j < nv; ++j) {
                st_ptr[nq + j] = d->qvel[j];
            }

            try {
                for (int t = 0; t < horizon; ++t) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                    if (elapsed > cutoff_time_) {
                        for (int remaining_t = t; remaining_t < horizon; ++remaining_t) {
                            for (int j = 0; j < nq; ++j) {
                                st_ptr[(remaining_t + 1) * nstate + j] = d->qpos[j];
                            }
                            for (int j = 0; j < nv; ++j) {
                                st_ptr[(remaining_t + 1) * nstate + nq + j] = d->qvel[j];
                            }
                            wbc_sensor_values_cpp(m, d, ids, policy_state, se_ptr + remaining_t * WBC_SENSOR_DIM);
                        }
                        break;
                    }

                    std::array<double, WBC_CONTROL_DIM> control{};
                    for (int j = 0; j < WBC_CONTROL_DIM; ++j) {
                        control[j] = controls_unchecked(i, t, j);
                    }
                    double* qvel = reference_qvel.data() + static_cast<size_t>(t) * 35;
                    apply_wbc_policy_step(m, d, ref_data, ids, policy_.get(), control.data(), qvel, policy_state);
                    mj_step(m, d);
                    for (int j = 0; j < WBC_CONTROL_DIM; ++j) {
                        policy_state.last_control[j] = static_cast<float>(control[j]);
                    }

                    for (int j = 0; j < nq; ++j) {
                        st_ptr[(t + 1) * nstate + j] = d->qpos[j];
                    }
                    for (int j = 0; j < nv; ++j) {
                        st_ptr[(t + 1) * nstate + nq + j] = d->qvel[j];
                    }
                    wbc_sensor_values_cpp(m, d, ids, policy_state, se_ptr + t * WBC_SENSOR_DIM);
                }
            } catch (...) {
                mj_deleteData(ref_data);
                throw;
            }

            mj_deleteData(ref_data);
        };

        if (num_threads_ == 0) {
            for (int i = 0; i < B; ++i) {
                execute_work(i);
            }
        } else {
            thread_pool_->execute_parallel(execute_work, B);
        }
    }

    auto states_arr = make_array_owned_g1(states_buf, B, horizon + 1, nstate);
    auto sensors_arr = make_array_owned_g1(sensor_buf, B, horizon, WBC_SENSOR_DIM);
    return py::make_tuple(states_arr, sensors_arr);
}

G1WBCRollout* G1WBCRollout::__enter__() {
    return this;
}

void G1WBCRollout::__exit__(py::object exc_type, py::object exc_val, py::object exc_tb) {
    close();
}

int G1WBCRollout::get_num_threads() const {
    return num_threads_;
}

py::array_t<float> SimG1WBC(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& command,
    const py::array_t<float>& prev_policy_state,
    const std::string& policy_path,
    const py::object& reference_qvel
) {
    OnnxPolicy* policy = cached_single_step_wbc_policy(policy_path);

    int nq = model->nq;
    int nv = model->nv;
    if (nq != WBC_CONTROL_DIM || nv != 35 || model->nu != WBC_ACTION_DIM) {
        throw std::runtime_error("G1 WBC native sim expects nq=36, nv=35, nu=29");
    }
    if (x0.ndim() != 1 || x0.shape(0) != nq + nv) {
        throw std::runtime_error("x0 must be a 1D array of shape (nq+nv)");
    }
    if (command.ndim() != 1 || command.shape(0) != WBC_CONTROL_DIM) {
        throw std::runtime_error("command must be a 1D array of shape 36");
    }

    WBCPolicyState policy_state;
    state_from_array(prev_policy_state, policy_state);

    const double* x0_ptr = x0.data();
    const double* control = command.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);

    mjData* ref_data = mj_makeData(model);
    if (ref_data == nullptr) {
        throw std::runtime_error("Failed to allocate MuJoCo reference data for G1 WBC sim");
    }
    WBCBodyIds ids = resolve_wbc_body_ids(model);

    double ref_qvel[35];
    if (qvel_from_optional_array(reference_qvel, nv, ref_qvel)) {
        // Explicit motion qvel matches tracking_bfm's motion command fields.
    } else if (policy_state.initialized) {
        std::array<double, WBC_CONTROL_DIM> previous_control{};
        for (int i = 0; i < WBC_CONTROL_DIM; ++i) {
            previous_control[i] = static_cast<double>(policy_state.last_control[i]);
        }
        qvel_from_control_pair(previous_control.data(), control, model->opt.timestep, ref_qvel);
        overwrite_policy_ang_vel_from_control_pair(
            model,
            ref_data,
            ids,
            previous_control.data(),
            control,
            model->opt.timestep,
            ref_qvel
        );
    } else {
        std::fill(ref_qvel, ref_qvel + 35, 0.0);
    }

    try {
        apply_wbc_policy_step(model, data, ref_data, ids, policy, control, ref_qvel, policy_state);
        mj_step(model, data);
    } catch (...) {
        mj_deleteData(ref_data);
        throw;
    }
    mj_deleteData(ref_data);

    for (int i = 0; i < WBC_CONTROL_DIM; ++i) {
        policy_state.last_control[i] = static_cast<float>(control[i]);
    }

    return state_to_array(policy_state);
}

py::dict DebugG1WBCPolicyStep(
    const mjModel* model,
    mjData* data,
    const py::array_t<double>& x0,
    const py::array_t<double>& command,
    const py::array_t<float>& prev_policy_state,
    const std::string& policy_path,
    const py::object& reference_qvel
) {
    OnnxPolicy* policy = cached_single_step_wbc_policy(policy_path);

    int nq = model->nq;
    int nv = model->nv;
    if (nq != WBC_CONTROL_DIM || nv != 35 || model->nu != WBC_ACTION_DIM) {
        throw std::runtime_error("G1 WBC native debug expects nq=36, nv=35, nu=29");
    }
    if (x0.ndim() != 1 || x0.shape(0) != nq + nv) {
        throw std::runtime_error("x0 must be a 1D array of shape (nq+nv)");
    }
    if (command.ndim() != 1 || command.shape(0) != WBC_CONTROL_DIM) {
        throw std::runtime_error("command must be a 1D array of shape 36");
    }

    WBCPolicyState policy_state;
    state_from_array(prev_policy_state, policy_state);

    const double* x0_ptr = x0.data();
    const double* control = command.data();
    mj_setState(model, data, x0_ptr, mjSTATE_QPOS | mjSTATE_QVEL);
    mj_forward(model, data);

    double ref_qvel[35];
    if (!qvel_from_optional_array(reference_qvel, nv, ref_qvel)) {
        std::fill(ref_qvel, ref_qvel + 35, 0.0);
    }

    mjData* ref_data = mj_makeData(model);
    if (ref_data == nullptr) {
        throw std::runtime_error("Failed to allocate MuJoCo reference data for G1 WBC debug");
    }

    std::vector<float> obs;
    std::vector<float> policy_out;
    try {
        WBCBodyIds ids = resolve_wbc_body_ids(model);
        build_wbc_observation(model, data, ref_data, ids, control, ref_qvel, policy_state, obs);
        policy_out = policy->run(obs);
        if (static_cast<int>(policy_out.size()) != WBC_ACTION_DIM) {
            throw std::runtime_error("G1 WBC policy output dimension must be 29");
        }
        for (int i = 0; i < WBC_ACTION_DIM; ++i) {
            policy_state.last_action[i] = policy_out[i];
            policy_state.held_ctrl[i] = static_cast<float>(policy_out[i] * WBC_ACTION_SCALE[i] + WBC_DEFAULT_JOINT_POS[i]);
            data->ctrl[i] = static_cast<double>(policy_state.held_ctrl[i]);
        }
    } catch (...) {
        mj_deleteData(ref_data);
        throw;
    }
    mj_deleteData(ref_data);

    for (int i = 0; i < WBC_CONTROL_DIM; ++i) {
        policy_state.last_control[i] = static_cast<float>(control[i]);
    }
    policy_state.step_count += WBC_POLICY_DECIMATION;

    py::dict result;
    result["policy_state"] = state_to_array(policy_state);
    result["observation"] = copy_to_float_array(obs.data(), WBC_OBS_DIM);
    result["action"] = copy_to_float_array(policy_out.data(), WBC_ACTION_DIM);
    result["held_ctrl"] = copy_to_float_array(policy_state.held_ctrl.data(), WBC_ACTION_DIM);
    return result;
}
