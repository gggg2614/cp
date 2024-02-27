<template>
  <div>
    <el-upload
      class="upload-demo"
      drag
      action="http://localhost:5000/upload"
      :show-file-list="false"
      :on-success="uploadSuccess"
      :before-upload="beforeUpload"
      :disabled="uploadDisabled"
    >
      <i class="el-icon-upload"></i>
      <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
    </el-upload>
    <el-button type="primary" @click="submit" :disabled="submitDisabled">{{ submitText }}</el-button>
    <el-loading v-if="loading" text="处理中..." fullscreen></el-loading>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive } from 'vue';
import axios from 'axios';
import { ElUpload, ElButton, ElMessage, ElLoading } from 'element-plus';

const files = ref([]);
const loading = ref(false);
const submitText = ref('开始特征选择');
const uploadDisabled = ref(false);
const submitDisabled = ref(true);

const uploadSuccess = (res, file, filelist) => {
  files.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage({ type: "success", message: "上传成功" });
  submitDisabled.value = false; // 允许点击开始特征选择按钮
};

const beforeUpload = (file: File) => {
  if (file.name.split(".")[1] !== "xlsx") {
    ElMessage.error("请上传xlsx");
    return false;
  }
};

const submit = async () => {
  loading.value = true; // 显示加载状态
  submitText.value = '处理中...'; // 更改按钮文本
  submitDisabled.value = true; // 禁用按钮
  if (!files.value || files.value.length === 0) {
    ElMessage.error('请上传文件。');
    return;
  }
  const formData = new FormData();
  formData.append('file', files.value[0]);
  try {
    const response = await fetch('http://localhost:5000/feature_selection', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    const data = await response.blob();
    downloadFile(data);
  } catch (error) {
    console.error('发生了与 fetch 操作相关的问题：', error);
    loading.value = false; // 隐藏加载状态
    submitText.value = '开始特征选择'; // 恢复按钮文本
    submitDisabled.value = false; // 允许点击按钮
  }
};

const downloadFile = (data: Blob) => {
  const blob = new Blob([data], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', 'results.csv');
  document.body.appendChild(link);
  link.click();
  loading.value = false; // 隐藏加载状态
  submitText.value = '开始特征选择'; // 恢复按钮文本
  submitDisabled.value = false; // 允许点击按钮
};

</script>

<style>
.upload-demo {
  margin-bottom: 20px;
}
</style>
