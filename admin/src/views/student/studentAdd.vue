<template>
  <div>
    <el-steps :active="activeStep" finish-status="success" align-center style="margin-bottom: 20px;">
      <el-step title="第一步"></el-step>
      <el-step title="第二步"></el-step>
      <el-step title="第三步"></el-step>
    </el-steps>

    <el-card v-if="activeStep === 0">
      <el-form label-width="120px" :model="form">
        <el-form-item label="输入搜索次数" prop="nSearches">
          <el-input v-model.trim="nSearches" placeholder="输入搜索次数" type="number" min="0"></el-input>
        </el-form-item>
        <el-form-item label="符号集" prop="nSearches">
          <el-input v-model.trim="functionSet" placeholder="输入搜索次数" ></el-input>
        </el-form-item>
        <el-form-item label="种群大小" prop="populationSize">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="populationSizeMin" placeholder="最小值" type="number" min="40"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="populationSizeMax" placeholder="宽度" type="number" min="40"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            对于小数据集来说，相对较小的种群就可以覆盖解空间
          </div>
        </el-form-item>

        <el-form-item label="迭代代数" prop="generations">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="generationsMin" placeholder="最小值" type="number" min="50"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="generationsMax" placeholder="宽度" type="number" min="50"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999; ">
            设置一个稍高的迭代次数可以深化算法的探索与优化
          </div>
        </el-form-item>

        <el-form-item label="简约系数" prop="parsimonyCoefficient">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="parsimonycoefficientMin" placeholder="最小值" type="number" min="0.000001"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="parsimonycoefficientMax" placeholder="宽度" type="number"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            简约系数用于避免过于复杂的解。考虑到数据规模较小，可以设置一个中等的值，如0.01到01，以鼓励算法寻找更简洁的解。
          </div>
        </el-form-item>

        <el-form-item label="最大样本数" prop="maxSamples">
          <el-col :span="5.5">
            <el-input v-model.trim="maxSamples" placeholder="最大样本数" type="number" min="1"></el-input>
          </el-col>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            由于数据集规模较小，可以考虑使用较大的样本比例，甚至是100%，以充分利用所有可用数据。
          </div>
        </el-form-item>

        <el-form-item label="交叉概率" prop="pCrossover">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="pCrossoverMin" placeholder="最小值" type="number" min="0.3" max="0.5"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="pCrossoverMax" placeholder="宽度" type="number" min="0.3" max="0.5"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            高交叉率有助于增加种群的多样性
          </div>
        </el-form-item>

        <el-form-item label="子树变异概率" prop="pSubtreeMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="pSubtreeMutationMin" placeholder="最小值" type="number" min="0.1"
                max="0.25"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="pSubtreeMutationMax" placeholder="宽度" type="number" min="0.1" max="0.25"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            变异率可以设置得相对较高，以促进解空间的探索
          </div>
        </el-form-item>

        <el-form-item label="提升变异" prop="pHoistMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="pHoistMutationMin" placeholder="最小值" type="number" min="0.05" max="0.15"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="pHoistMutationMax" placeholder="宽度" type="number" min="0.05" max="0.15"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            由于提升变异可以帮助简化解的结构，推荐的概率设置为0.05到0.1之间。
          </div>
        </el-form-item>

        <el-form-item label="点变异" prop="pPointMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input v-model.trim="pPointMutationMin" placeholder="最小值" type="number" min="0.05" max="0.15"></el-input>
            </el-col>
            <el-col :span="12">
              <el-input v-model.trim="pPointMutationMax" placeholder="宽度" type="number" min="0.05" max="0.15"></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999;">
            点变异直接修改程序树的细节，建议的概率也是0.05到0.1之间。
          </div>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="nextStep">下一步</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-upload v-if="activeStep === 1" drag :action="importURL" :on-success="uploadSuccess" :before-upload="beforeUpload"
      :auto-upload="true">
      <el-button slot="trigger" type="primary">选择文件</el-button>
      <div class="el-upload__tip" slot="tip">支持上传 XLSX/XLS 文件</div>
    </el-upload>

    <el-card v-if="activeStep === 2">
      <h3>返回的结果：</h3>
      <el-form label-width="200" v-if="Object.keys(form).length > 0">
        <el-row justify="center">
          <el-col v-for="(value, key) in form" :key="key">
            <el-form-item :label="key">
              <el-input v-model.trim="form[key]" :readonly="true"></el-input>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
      <el-button @click="prevStep" type="primary">返回上一步</el-button>
      <el-button @click="downloadCSV" type="primary" :loading="csvLoading">下载结果</el-button>
      <el-button type="primary" size="default" @click="downloadTxt">下载转换前的特征</el-button>
      <el-button type="primary" size="default" @click="downloadtxtNames">下载转换后的特征</el-button>
    </el-card>
    
    <el-card v-if="activeStep === 3">
      <h3>生成的特征：</h3>
      <div v-if="gpFeatures.length > 0">
        <ul>
          <li v-for="(feature, index) in gpFeatures" :key="index">{{ feature }}</li>
        </ul>
      </div>
      <div v-else>
        <p>暂无特征数据。</p>
      </div>
    </el-card>


    <el-button v-if="activeStep === 1" @click="submit" type="primary" :loading="subLoading">提交</el-button>
    <el-button v-show="activeStep === 1" @click="prevStep" slot="tip">返回</el-button>
  </div>
</template>

<script lang="ts" setup>
import { ref, nextTick } from 'vue';
import { ElMessage } from 'element-plus';

const activeStep = ref(0);
const nSearches = ref('1');
const files = ref([]);
const form = ref({});
const subLoading = ref(false);
const csvLoading = ref(false);
const importURL = 'http://localhost:5000/upload';
const populationSizeMin = ref(20);
const populationSizeMax = ref(0);
const generationsMin = ref(1);
const generationsMax = ref(1);
const maxSamples = ref('1');
const pCrossoverMin = ref(0.3);
const pCrossoverMax = ref(0.2);
const pSubtreeMutationMin = ref(0.1);
const pSubtreeMutationMax = ref(0.15);
const pHoistMutationMin = ref(0.05);
const pHoistMutationMax = ref(0.1);
const pPointMutationMin = ref(0.05);
const pPointMutationMax = ref(0.1);
const parsimonycoefficientMin = ref(0.000001);
const parsimonycoefficientMax = ref(0.099999);
const functionSet = ref(['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'])
const gpFeatures = ref('')

const handleBackendResponse = (data) => {
  const bestParams = data.best_params;
  form.value = {}; // 清空 form
  for (const key in bestParams) {
    form.value[key] = bestParams[key];
  }
  form.value['best_score'] = data.best_score;
};

const downloadTxt = async()=>{
  try {
    const response = await fetch('http://localhost:5000/download_txt', {
      method: 'GET',
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'symbolic_transformer_formulas.txt');
    document.body.appendChild(link);
    link.click();
    ElMessage.success('txt downloaded successfully');
  } catch (error) {
    console.error('Error:', error);
    ElMessage.error('Failed to download model txt');
  }  
}

const downloadtxtNames = async()=>{
  try {
    const response = await fetch('http://localhost:5000/download_txt_names', {
      method: 'GET',
    });
    const data = await response.blob();
    const url = window.URL.createObjectURL(data);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'symbolic_transformer_formulas_with_feature_names.txt');
    document.body.appendChild(link);
    link.click();
    ElMessage.success('txtNames downloaded successfully');
  } catch (error) {
    console.error('Error:', error);
    ElMessage.error('Failed to download model txtNames');
  }
}

const forthStep = async()=>{
  const formData = new FormData();
  formData.append('file', files.value[0]);
  formData.append('best_params',JSON.stringify(form.value))
  fetch('http://localhost:5000/get_gp_features', {
    method: 'POST',
    body: formData, // 将前端获取的 form 传给后端
  }).then(response => {
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    return response.json();
  })
}

const nextStep = () => {
  if (parseInt(nSearches.value) <= 0) {
    ElMessage.error('请输入大于0的整数')
    return;
  }
  (async () => {
    await nextTick();
    if (nSearches.value.trim() === '') {
      ElMessage.error('请输入搜索次数。');
      return;
    }
    if (!/^\d+$/.test(nSearches.value)) {
      ElMessage.error('搜索次数必须为整数。');
      return;
    }
    activeStep.value++;
  })();
};

const prevStep = () => {
  activeStep.value--;
};

const beforeUpload = (file: File) => {
  if (file.name.split(".")[1] !== "xlsx") {
    ElMessage.error("请上传xlsx");
    return false;
  }
};

const uploadSuccess = (res, file, filelist) => {
  files.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage({ type: "success", message: "上传成功" });
};

const downloadCSV = () => {
  fetch('http://localhost:5000/generate_features', {
    method: 'POST',
    body: JSON.stringify({ best_params: form.value }), // 将前端获取的 form 传给后端
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(response => {
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    return response.blob();
  }).then(blob => {
    const url = window.URL.createObjectURL(new Blob([blob]));
    const link = document.createElement('a');
    csvLoading.value = true;
    link.href = url;
    link.setAttribute('download', 'features.csv'); // 设置下载的文件名为 features.csv
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  }).catch(error => {
    console.error('发生了与 fetch 操作相关的问题：', error);
  }).finally(() => {
    csvLoading.value = false;
  });
};

const submit = () => {
  if (!files.value || files.value.length === 0) {
    ElMessage.error('请上传文件。');
    return;
  }
  const formData = new FormData();
  formData.append('file', files.value[0]);
  formData.append('n_searches', nSearches.value);
  formData.append('population_size', `${populationSizeMin.value},${populationSizeMax.value}`);
  formData.append('generations', `${generationsMin.value},${generationsMax.value}`);
  formData.append('maxSamples', maxSamples.value);
  formData.append('pCrossover', `${pCrossoverMin.value},${pCrossoverMax.value}`);
  formData.append('pSubtreeMutation', `${pSubtreeMutationMin.value},${pSubtreeMutationMax.value}`);
  formData.append('pHoistMutation', `${pHoistMutationMin.value},${pHoistMutationMax.value}`);
  formData.append('pPointMutation', `${pPointMutationMin.value},${pPointMutationMax.value}`);
  formData.append('parsimonyCoefficient', `${parsimonycoefficientMin.value},${parsimonycoefficientMax.value}`);
  formData.append('functionSet',functionSet.value);
  subLoading.value = true;
  forthStep()
  fetch('http://localhost:5000/file1', {
    method: 'POST',
    body: formData
  }).then(response => {
    if (!response.ok) {
      throw new Error('网络响应不正常');
    }
    return response.json();
  }).then(data => {
    handleBackendResponse(data);
    activeStep.value = 2;
  }).catch(error => {
    console.error('发生了与 fetch 操作相关的问题：', error);
  }).finally(() => {
    subLoading.value = false;
  });

};
</script>
