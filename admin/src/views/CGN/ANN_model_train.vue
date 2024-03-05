<template>
  <div>
    <el-steps
      :active="activeStep"
      finish-status="success"
      align-center
      style="margin-bottom: 20px"
    >
      <el-step title="第一步"></el-step>
      <el-step title="第二步"></el-step>
    </el-steps>

    <el-container v-if="activeStep === 0">
      <el-main>
        <el-form ref="form" :model="formData" label-width="120px">
          <el-form-item label="Cycles">
            <el-input
              @input="handleInput('cycles')"
              v-model.trim="formData.cycles"
              placeholder="请输入整数"
              min="0"
              type="number"
              step="1"
            ></el-input>
          </el-form-item>
          <el-form-item label="Epoch">
            <el-input
              v-model.trim="formData.epoch"
              placeholder="请输入整数"
              min="0"
              type="number"
            ></el-input>
          </el-form-item>
          <el-form-item label="FeatureAddNum">
            <el-input
              v-model.trim="formData.feature_add_num"
              placeholder="请输入0-10的数"
              type="number"
              min="0"
              max="10"
            ></el-input>
          </el-form-item>
          <el-form-item label="Data File(.xlsx)">
            <el-upload
              action="/upload"
              :on-success="handleDataUpload"
              :before-upload="file => beforeUpload(file, 'xlsx')"
              :limit="1"
              :on-error="handleError"
            >
              <el-button slot=" trigger" type="primary">Upload</el-button
              >(最多上传一个文件)
            </el-upload>
          </el-form-item>
          <el-form-item label="Data2 File(.csv)">
            <el-upload
              :on-success="handleData2Upload"
              action="/upload"
              :before-upload="file => beforeUpload(file, 'csv')"
              :limit="1"
              :on-error="handleError"
            >
              <el-button slot="trigger" type="primary">Upload</el-button
              >(最多上传一个文件)
            </el-upload>
          </el-form-item>
          <el-form-item>
            <el-button
              type="primary"
              @click="validateAndTrain"
              :loading="trainLoading"
              :disabled="!isFormValid"
              >Train Model</el-button
            >
          </el-form-item>
          <el-form-item v-if="trainComplete">
            <el-button type="success" @click="downloadCsv"
              >Download CSV</el-button
            >
            <el-button type="success" @click="downloadPth"
              >Download PTH</el-button
            >
            <el-button type="success" @click="downloadjoblibX"
              >Download joblibX</el-button
            >
            <el-button type="success" @click="downloadjoblibY"
              >Download joblibY</el-button
            >
            <el-button type="primary" @click="nextStep">下一步</el-button>
          </el-form-item>
        </el-form>
      </el-main>
    </el-container>

    <el-container v-else-if="activeStep === 1">
      <el-main>
        <div>
          <el-upload
            action="/upload"
            :on-success="handlePthUpload"
            :auto-upload="true"
            :before-upload="file => beforeUpload(file, 'pth')"
            :limit="1"
            :on-error="handleError"
          >
            <template #default>
              模型文件(.pth)：<el-button type="primary">上传模型文件</el-button
              >(最多上传一个文件)
            </template>
          </el-upload>

          <el-upload
            action="/upload"
            :on-success="handlePreUpload"
            :limit="1"
            :auto-upload="true"
            :before-upload="file => beforeUpload(file, 'xlsx')"
            :on-error="handleError"
          >
            <template #default>
              数据文件(.xlsx)：<el-button type="primary">上传数据文件</el-button
              >(最多上传一个文件)
            </template>
          </el-upload>
          <el-button @click="predict" :loading="loading" type="primary"
            >预测</el-button
          >
          <el-button v-if="fileOK" type="success" @click="downloadPreFile"
            >下载预测结果</el-button
          >
          <el-button type="primary" @click="activeStep--">返回上一步</el-button>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { ElMessage } from "element-plus";

const activeStep = ref(0);
const formData = ref({
  cycles: "",
  epoch: "",
  feature_add_num: ""
});
const data = ref([]);
const data2 = ref([]);
const trainLoading = ref(false);
const trainComplete = ref(false);
const modelFile = ref([]);
const dataFile = ref([]);
const loading = ref(false);
const fileOK = ref(false);

const handleInput = (key) => (event) => {
  const input = event.target.value;
  formData.value[key] = input.replace(/\D/g, ''); // 删除所有非数字字符
};
const predict = async () => {
  if (modelFile.value.length === 0 || dataFile.value.length === 0) {
    ElMessage.error("请上传文件");
    return;
  }

  loading.value = true;

  const formData = new FormData();
  formData.append("model", modelFile.value[0]); // 使用 'file' 作为模型文件的键
  formData.append("data_pre", dataFile.value[0]); // 使用 'file' 作为数据文件的键

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    fileOK.value = true;
    return response

  } catch (error) {
    ElMessage.error('发生错误')
  } finally {
    loading.value = false;
  }
};


const validateAndTrain = () => {
  const featureAddNum = parseInt(formData.value.feature_add_num);
  if (isNaN(featureAddNum) || featureAddNum < 0 || featureAddNum > 10) {
    console.log(featureAddNum);
    ElMessage.error("Feature Add Num must be a number between 0 and 10");
    return;
  }
  if (data.value.length === 0 || data2.value.length === 0) {
    ElMessage.error("请上传文件");
    return;
  }
  trainModel();
};

const nextStep = () => {
  activeStep.value++;
};

const handlePthUpload = (response: any, file, filelist) => {
  modelFile.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("pth file uploaded successfully");
};
const handlePreUpload = (response: any, file, filelist) => {
  dataFile.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("model file uploaded successfully");
};
const handleDataUpload = (response: any, file, filelist) => {
  data.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("Data file uploaded successfully");
};
const handleData2Upload = (response: any, file, filelist) => {
  data2.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("Data2 file uploaded successfully");
};

const trainModel = async () => {
  if (!isFormValid) return;
  trainLoading.value = true;
  const formData1 = new FormData();
  formData1.append("cycles", formData.value.cycles);
  formData1.append("epoch", formData.value.epoch);
  formData1.append("feature_add_num", formData.value.feature_add_num);
  formData1.append("data", data.value[0]);
  formData1.append("data2", data2.value[0]);
  try {
    const response = await fetch("/train", {
      method: "POST",
      body: formData1
    });
    const data = await response;
    if (data.status == 200) {
      ElMessage.success("Model trained successfully");
      trainComplete.value = true;
    } else {
      ElMessage.error("训练失败，请检查文件是否有误");
    }
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("训练失败，请检查文件是否有误");
  } finally {
    trainLoading.value = false;
  }
};

const beforeUpload = (file: File, expectedExtension: string) => {
  const extension = file.name.split(".")[1];
  if (extension !== expectedExtension) {
    ElMessage.error(`请上传.${expectedExtension}文件`);
    return false;
  }
};

const downloadCsv = async () => {
  try {
    const response = await fetch("/download_results", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "results.csv");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("Results CSV downloaded successfully");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download results CSV");
  }
};

const downloadPth = async () => {
  try {
    const response = await fetch("/download_model", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "best_model.pth");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("Model PTH downloaded successfully");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model PTH");
  }
};

const downloadjoblibX = async () => {
  try {
    const response = await fetch("/download_joblibX", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "X_scaler.joblib");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("Model PTH downloaded successfully");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model PTH");
  }
};
const downloadjoblibY = async () => {
  try {
    const response = await fetch("/download_joblibY", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "Y_scaler.joblib");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("Model PTH downloaded successfully");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model PTH");
  }
};

const downloadPreFile = async () => {
  try {
    const response = await fetch("/download_predict_file", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "predicted_output_file.xlsx");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("predicted file downloaded successfully");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model file");
  }
};
const isFormValid = computed(() => {
  return (
    formData.value.cycles &&
    formData.value.epoch &&
    formData.value.feature_add_num
  );
});
const handleError = (error, file, fileList) => {
  ElMessage.error('上传文件发生错误，请检查文件是否正确');
  return;
};
</script>
