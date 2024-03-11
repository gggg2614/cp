<template>
  <div v-loading="vloading" element-loading-text="处理中...">
    <el-container>
      <el-main>
        <el-form ref="form" :model="formData" label-width="120px">
          <el-form-item label="需要训练的轮数">
            <el-input
              @input="handleInput('cycles')"
              v-model.trim="formData.cycles"
              placeholder="请输入整数"
              min="0"
              type="number"
              step="1"
            ></el-input>
          </el-form-item>
          <el-form-item label="每轮训练的次数">
            <el-input
              v-model.trim="formData.epoch"
              placeholder="请输入整数"
              min="0"
              type="number"
            ></el-input>
          </el-form-item>
          <el-form-item label="添加的特征个数">
            <el-input
              v-model.trim="formData.feature_add_num"
              placeholder="请输入0-10的数"
              type="number"
              min="0"
              max="10"
            ></el-input>
          </el-form-item>
          <el-form-item label-width="115" label="原始数据(.xlsx)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handleDataUpload"
              :before-upload="file => beforeUpload(file, 'xlsx')"
              :limit="1"
              :on-error="handleError"
              :on-remove="()=>uploadRemove('data')"
              :file-list="data"
              >
              <el-button slot=" trigger" type="primary">Upload</el-button
                >(最多上传一个文件)
              </el-upload>
            </el-form-item>
            <el-form-item label-width="280" label="符号回归特征构建中得到的数据文件(.csv)">
            <el-upload
              :on-success="handleData2Upload"
              action="http://localhost:5000/upload"
              :before-upload="file => beforeUpload(file, 'csv')"
              :limit="1"
              :on-remove="()=>uploadRemove('data2')"
              :on-error="handleError"
              :file-list="data2"
            >
              <el-button slot="trigger" type="primary">Upload</el-button
              >(最多上传一个文件)
            </el-upload>
          </el-form-item>
            <el-button
              type="primary"
              @click="validateAndTrain"
              :loading="trainLoading"
              :disabled="!isFormValid"
              >训练模型</el-button
            >
          <el-form-item v-if="trainComplete">
            <el-button type="success" @click="downloadCsv"
              >下载各轮次模型表现</el-button
            >
            <el-button type="success" @click="downloadPth"
              >下载模型参数文件</el-button
            >
            <el-button type="success" @click="downloadjoblibX"
              >下载输入特征归一化参数</el-button
            >
            <el-button type="success" @click="downloadjoblibY"
              >下载输出特征归一化参数</el-button
            >
          </el-form-item>
        </el-form>
      </el-main>
    </el-container>

  </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { ElMessage } from "element-plus";

const activeStep = ref(0);
const formData = ref({
  cycles: "200",
  epoch: "3000",
  feature_add_num: "6"
});
const data = ref([]);
const data2 = ref([]);
const trainLoading = ref(false);
const trainComplete = ref(false);
const loading = ref(false);
const fileOK = ref(false);
const vloading = ref(false);

const handleInput = (key) => (event) => {
  const input = event.target.value;
  formData.value[key] = input.replace(/\D/g, ''); // 删除所有非数字字符
};

const validateAndTrain = () => {
  const featureAddNum = parseInt(formData.value.feature_add_num);
  if (isNaN(featureAddNum) || featureAddNum < 0 || featureAddNum > 10) {
    console.log(featureAddNum);
    ElMessage.error("特征个数必须在0-10之间");
    return;
  }
  if (data.value.length === 0 || data2.value.length === 0) {
    ElMessage.error("请上传文件");
    return;
  }
  trainModel();
};


const handleDataUpload = (response: any, file, filelist) => {
  data.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("原始数据上传成功");
};
const handleData2Upload = (response: any, file, filelist) => {
  data2.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("数据文件上传成功");
};

const trainModel = async () => {
  if (!isFormValid) return;
  trainLoading.value = true;
  vloading.value = true
  const formData1 = new FormData();
  formData1.append("cycles", formData.value.cycles);
  formData1.append("epoch", formData.value.epoch);
  formData1.append("feature_add_num", formData.value.feature_add_num);
  formData1.append("data", data.value[0]);
  formData1.append("data2", data2.value[0]);
  try {
    const response = await fetch("http://localhost:5000/train", {
      method: "POST",
      body: formData1
    });
    const data = await response;
    if (data.status == 200) {
      ElMessage.success("模型训练成功");
      trainComplete.value = true;
    } else {
      ElMessage.error("训练失败，请检查文件是否有误");
    }
  } catch (error) {
    ElMessage.error("训练失败，请检查文件是否有误");
  } finally {
    trainLoading.value = false;
    vloading.value = false;
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
    const response = await fetch("http://localhost:5000/download_results", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "各轮次模型表现.csv");
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
    link.setAttribute("download", "模型参数文件.pth");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("pth文件下载成功");
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
    ElMessage.success("joblibx下载成功");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("下载失败");
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
    ElMessage.success("jobliby下载成功");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("下载失败");
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
const uploadRemove = (fileType:string) => {
  if (fileType === "data") {
    data.value = [];
  } else if (fileType === "data2") {
    data2.value = [];
  }
  else{
    return;
  }
}

</script>
