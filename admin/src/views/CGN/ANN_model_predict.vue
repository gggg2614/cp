<template>
  <div v-loading="vloading" element-loading-text="处理中...">
    <el-container>
      <el-main>
        <el-form  label-width="185px">
          <el-form-item label="输入特征归一化参数(.joblib)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handleXUpload"
              :auto-upload="true"
              :before-upload="file => beforeUpload(file, 'joblib')"
              :limit="1"
              :on-error="handleError"
              :on-remove="() => uploadRemove('xJoblib')"
              :file-list="xJoblib"
            >
              <el-button slot="trigger" type="primary">上传模型文件</el-button>(最多上传一个文件)
            </el-upload>
          </el-form-item>

          <el-form-item label="输出特征归一化参数(.joblib)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handleYUpload"
              :auto-upload="true"
              :before-upload="file => beforeUpload(file, 'joblib')"
              :limit="1"
              :on-error="handleError"
              :on-remove="() => uploadRemove('yJoblib')"
              :file-list="yJoblib"
            >
              <el-button slot="trigger" type="primary">上传模型文件</el-button>(最多上传一个文件)
            </el-upload>
          </el-form-item>

          <el-form-item label="模型文件(.pth)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handlePthUpload"
              :auto-upload="true"
              :before-upload="file => beforeUpload(file, 'pth')"
              :limit="1"
              :on-error="handleError"
              :on-remove="() => uploadRemove('modelFile')"
              :file-list="modelFile"
            >
              <el-button slot="trigger" type="primary">上传模型文件</el-button>(最多上传一个文件)
            </el-upload>
          </el-form-item>

          <el-form-item label="数据文件(.xlsx)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handlePreUpload"
              :limit="1"
              :auto-upload="true"
              :before-upload="file => beforeUpload(file, 'xlsx')"
              :on-error="handleError"
              :on-remove="() => uploadRemove('dataFile')"
              :file-list="dataFile"
            >
              <el-button slot="trigger" type="primary">上传数据文件</el-button>(最多上传一个文件)
            </el-upload>
          </el-form-item>

          <el-form-item label="特征公式(.txt)">
            <el-upload
              action="http://localhost:5000/upload"
              :on-success="handleTxtUpload"
              :limit="1"
              :auto-upload="true"
              :before-upload="file => beforeUpload(file, 'txt')"
              :on-error="handleError"
              :on-remove="() => uploadRemove('tzTxt')"
              :file-list="tzTxt"
            >
              <el-button slot="trigger" type="primary">上传特征公式</el-button>(最多上传一个文件)
            </el-upload>
          </el-form-item>

          <el-form-item>
            <el-button @click="predict" :loading="loading" type="primary">预测</el-button>
            <el-button v-if="fileOK" type="success" @click="downloadPreFile">下载预测结果</el-button>
          </el-form-item>
        </el-form>
      </el-main>
    </el-container>
  </div>
</template>

  <script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";

const modelFile = ref([]);
const dataFile = ref([]);
const xJoblib = ref([]);
const yJoblib = ref([]);
const tzTxt = ref([]);
const loading = ref(false);
const vloading = ref(false);
const fileOK = ref(false);

const handleXUpload = (response: any, file, filelist) => {
  xJoblib.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("输入特征归一化参数文件上传成功");
};

const handleYUpload = (response: any, file, filelist) => {
  yJoblib.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("输出特征归一化参数文件上传成功");
};

const handlePthUpload = (response: any, file, filelist) => {
  modelFile.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("模型文件上传成功");
};

const handlePreUpload = (response: any, file, filelist) => {
  dataFile.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("模型文件上传成功");
};

const handleTxtUpload = (response: any, file, filelist) => {
  tzTxt.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage.success("特征公式文件上传成功");
};

const predict = async () => {
  if (
    modelFile.value.length === 0 ||
    dataFile.value.length === 0 ||
    xJoblib.value.length === 0 ||
    yJoblib.value.length === 0 ||
    tzTxt.value.length === 0
  ) {
    ElMessage.error("请上传文件");
    return;
  }
  loading.value = true;
  vloading.value = true;
  const formData = new FormData();
  formData.append("model", modelFile.value[0]); // 使用 'file' 作为模型文件的键
  formData.append("data_pre", dataFile.value[0]); 
  formData.append("x_joblib", xJoblib.value[0]); 
  formData.append("y_joblib", yJoblib.value[0]); 
  formData.append("tzTxt", tzTxt.value[0]); 
  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData
    });
    if (!response.ok) {
      const errorMessage = await response.json(); // 解析返回的 JSON 数据
      throw new Error(`${errorMessage.error}`);
    }
    fileOK.value = true;
  } catch (error) {
    ElMessage.error("发生错误: " + error.message);
  } finally {
    loading.value = false;
    vloading.value = false;
  }
};

const downloadPreFile = async () => {
  try {
    const response = await fetch("http://localhost:5000/download_predict_file", {
      method: "GET"
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "预测结果.xlsx");
    document.body.appendChild(link);
    link.click();
    ElMessage.success("下载成功");
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("下载失败");
  }
};

const beforeUpload = (file: File, expectedExtension: string) => {
  const extension = file.name.split(".")[1];
  if (extension !== expectedExtension) {
    ElMessage.error(`请上传.${expectedExtension}文件`);
    return false;
  }
};

const handleError = (error, file, fileList) => {
  ElMessage.error("上传文件发生错误，请检查文件是否正确");
};

const uploadRemove = (fileType:string) => {
  if (fileType === "xJoblib") {
    xJoblib.value = [];
  } else if (fileType === "yJoblib") {
    yJoblib.value = [];
  }else if (fileType === "modelFile"){
    modelFile.value = [];
  }else if(fileType === "dataFile"){
    dataFile.value = []
  }else if(fileType === "tzTxt"){
    tzTxt.value = []
  }
  else{
    return;
  }
}
</script>
  