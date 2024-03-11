<template>
  <div v-loading="vloading" element-loading-text="处理中...">
    <el-steps
      :active="activeStep"
      finish-status="success"
      align-center
      style="margin-bottom: 20px"
    >
      <el-step title="第一步"></el-step>
      <el-step title="第二步"></el-step>
      <el-step title="下载结果"></el-step>
    </el-steps>

    <el-card v-if="activeStep === 0">
      <el-form label-width="120px" :model="form">
        <el-form-item label="输入搜索次数" prop="nSearches">
          <el-input
          v-model.trim="nSearches"
          placeholder="输入搜索次数"
          type="number"
          :controls = false
          min="0"
          ></el-input>
        </el-form-item>
        <el-form-item label="交叉验证折数" prop="cv">
          <el-input
          v-model.trim="cv"
          placeholder="输入交叉验证折数"
          type="number"
          :controls = false
          min="0"
          ></el-input>
        </el-form-item>
        <el-form-item label="符号集" prop="functionSet">
          <el-input
            v-model.trim="functionSet"
            placeholder="输入符号集"
          ></el-input>
        </el-form-item>
        <el-form-item label="种群大小" prop="populationSize">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="populationSizeMin"
                placeholder="最小值"
                type="number"
                min="40"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="populationSizeMax"
                placeholder="宽度"
                type="number"
                min="40"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            对于小数据集来说，相对较小的种群就可以覆盖解空间
          </div>
        </el-form-item>
        <el-form-item label="锦标赛规模" prop="tournamentSize">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="tournamentMin"
                placeholder="最小值"
                type="number"
                min="40"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="tournamentMax"
                placeholder="宽度"
                type="number"
                min="40"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            锦标赛”参加选手每轮参与竞选的个体，小值使得适应度较低的个体也有较高的机会被选中，这有助于保持种群的多样性，大值将优先选择较高适应度的个体
          </div>
        </el-form-item>
        <el-form-item label="迭代代数" prop="generations">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="generationsMin"
                placeholder="最小值"
                type="number"
                min="50"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="generationsMax"
                placeholder="宽度"
                type="number"
                min="50"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            设置一个稍高的迭代次数可以深化算法的探索与优化
          </div>
        </el-form-item>

        <el-form-item label="简约系数" prop="parsimonyCoefficient">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.number="parsimonycoefficientMin"
                placeholder="最小值"
                type="text"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="parsimonycoefficientMax"
                placeholder="宽度"
                type="text"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            简约系数用于避免过于复杂的解。考虑到数据规模较小，可以设置一个中等的值，如0.01到0.1，以鼓励算法寻找更简洁的解。
          </div>
        </el-form-item>
        <el-form-item label="随机状态" prop="randomstate">
            <el-row :gutter="20">
              <el-col :span="12">
                <el-input
                  v-model.trim="randomstateMin"
                  placeholder="最小值"
                  type="number"
                  min="40"
                ></el-input>
              </el-col>
              <el-col :span="12">
                <el-input
                  v-model.trim="randomstateMax"
                  placeholder="宽度"
                  type="number"
                  min="40"
                ></el-input>
              </el-col>
            </el-row>
            <div style="margin-top: 10px; font-size: 12px; color: #999">
              通常用于设置随机数生成器的种子，以确保结果的可重复性。
            </div>
          </el-form-item>
        <el-form-item label="最大样本数" prop="maxSamples">
          <el-col :span="5.5">
            <el-input
              v-model.trim="maxSamples"
              placeholder="最大样本数"
              type="number"
              min="1"
            ></el-input>
          </el-col>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            由于数据集规模较小，可以考虑使用较大的样本比例，甚至是100%，以充分利用所有可用数据。
          </div>
        </el-form-item>

        <el-form-item label="交叉概率" prop="pCrossover">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="pCrossoverMin"
                placeholder="最小值"
                type="number"
                min="0.3"
                max="0.5"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="pCrossoverMax"
                placeholder="宽度"
                type="number"
                min="0.3"
                max="0.5"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            高交叉率有助于增加种群的多样性
          </div>
        </el-form-item>

        <el-form-item label="子树变异概率" prop="pSubtreeMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="pSubtreeMutationMin"
                placeholder="最小值"
                type="number"
                min="0.1"
                max="0.25"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="pSubtreeMutationMax"
                placeholder="宽度"
                type="number"
                min="0.1"
                max="0.25"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            变异率可以设置得相对较高，以促进解空间的探索
          </div>
        </el-form-item>

        <el-form-item label="提升变异" prop="pHoistMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="pHoistMutationMin"
                placeholder="最小值"
                type="number"
                min="0.05"
                max="0.15"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="pHoistMutationMax"
                placeholder="宽度"
                type="number"
                min="0.05"
                max="0.15"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            由于提升变异可以帮助简化解的结构，推荐的概率设置为0.05到0.1之间。
          </div>
        </el-form-item>

        <el-form-item label="点变异" prop="pPointMutation">
          <el-row :gutter="20">
            <el-col :span="12">
              <el-input
                v-model.trim="pPointMutationMin"
                placeholder="最小值"
                type="number"
                min="0.05"
                max="0.15"
              ></el-input>
            </el-col>
            <el-col :span="12">
              <el-input
                v-model.trim="pPointMutationMax"
                placeholder="宽度"
                type="number"
                min="0.05"
                max="0.15"
              ></el-input>
            </el-col>
          </el-row>
          <div style="margin-top: 10px; font-size: 12px; color: #999">
            点变异直接修改程序树的细节，建议的概率也是0.05到0.1之间。
          </div>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="nextStep">下一步</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-upload
      v-if="activeStep === 1"
      drag
      :action="importURL"
      :on-success="uploadSuccess"
      :before-upload="beforeUpload"
      :on-remove="uploadRemove"
      :auto-upload="true"
      :limit="1"
      :on-error="handleError"
      :file-list="files"
    >
      <el-button slot="trigger" type="primary">选择文件</el-button>
      <div class="el-upload__tip" slot="tip">支持上传 XLSX/XLS 文件</div>
      (最多上传一个文件)
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
      <el-button @click="stepThird" type="primary" :loading="csvLoading"
        >下一步</el-button
      >
    </el-card>
    
    <el-card v-if="activeStep === 3">
      <div v-if="gpFeatures">
        <p>共构建出特征个数为：{{ gpFeatures.featureNum }}</p>
        <p>数学特征数：{{ gpFeatures.featureMathNum }}</p>
        <p>特征信息：{{ gpFeatures.featureInfo }}</p>
      </div>
      <div v-else>
        <p>暂无特征数据。</p>
      </div>
      <el-button @click="prevStep" type="primary">返回上一步</el-button>
      <el-button type="success" size="default" @click="downloadTxt"
        >下载转换前的特征</el-button
      >
      <el-button type="success" size="default" @click="downloadtxtNames"
        >下载转换后的特征</el-button
      >
      <el-button type="success" size="default" @click="downloadFinalCsv"
        >下载最终筛选特征值</el-button
      >
      <el-button type="success" size="default" @click="downloadTotalCsv"
        >下载初始构建特征值</el-button
      >
      <el-button type="success" size="default" @click="downloadxJob"
        >下载输入特征归一化参数</el-button
      >
      <el-button type="success" size="default" @click="downloadyJob"
        >下载输出变量归一化参数</el-button
      >
    </el-card>

    <el-button
      v-if="activeStep === 1"
      :disabled="files.length > 0 ? false : true"
      @click="submit"
      type="primary"
      :loading="subLoading"
      >提交</el-button
    >
    <el-button v-show="activeStep === 1" @click="prevStep" slot="tip"
      >返回</el-button
    >
  </div>
</template>

<script lang="ts" setup>
import { ref, nextTick, watch } from "vue";
import { ElMessage } from "element-plus";

const activeStep = ref(0);
const nSearches = ref("2");
const cv = ref('3')
const files = ref([]);
const form = ref({});
const subLoading = ref(false);
const csvLoading = ref(false);
const importURL = "http://localhost:5000/upload";
const populationSizeMin = ref(100);
const populationSizeMax = ref(200);
const generationsMin = ref(20);
const generationsMax = ref(50);
const maxSamples = ref([1]);
const pCrossoverMin = ref(0.1);
const pCrossoverMax = ref(0.3);
const pSubtreeMutationMin = ref(0.1);
const pSubtreeMutationMax = ref(0.3);
const pHoistMutationMin = ref(0.05);
const pHoistMutationMax = ref(0.2);
const pPointMutationMin = ref(0.05);
const pPointMutationMax = ref(0.2);
const parsimonycoefficientMin = ref('0.00000000001');
const parsimonycoefficientMax = ref(0.0001);
const tournamentMin = ref(1)
const tournamentMax = ref(20)
const randomstateMin = ref(1)
const randomstateMax = ref(1000)
const vloading = ref(false);
const functionSet = ref([
  "add",
  "sub",
  "mul",
  "div",
  "sqrt",
  "log",
  "abs",
  "neg",
  "inv"
]);
const gpFeatures = ref({
  featureInfo:'',
  featureMathNum:'',
  featureNum:''
});

const handleBackendResponse = data => {
  const bestParams = data.best_params;
  form.value = {}; // 清空 form
  for (const key in bestParams) {
    form.value[key] = bestParams[key];
  }
  form.value["best_score"] = data.best_score;
};

const downloadxJob = async() => {
  try {
    const response = await fetch("http://localhost:5000/download_file1_x_job", {
      method: "GET"
    });
    if(response.ok){
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "输入特征归一化参数.joblib");
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    }else{
      ElMessage.error('发生错误')
      return;
    }
  } 
  catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txt");
  }
}

const downloadyJob = async() => {
  try {
    const response = await fetch("http://localhost:5000/download_file1_y_job", {
      method: "GET"
    });
    if(response.ok){
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "输出变量归一化参数.joblib");
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    }else{
      ElMessage.error('发生错误')
      return;
    }
  } 
  catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txt");
  }
}

const downloadTotalCsv = async() => {
  try {
    const response = await fetch("http://localhost:5000/download_total_csv", {
      method: "GET"
    });
    if(response.ok){
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "初始构建特征值.csv");
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    }else{
      ElMessage.error('发生错误')
      return;
    }
  } 
  catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txt");
  }
}

const downloadFinalCsv = async() => {
  try {
    const response = await fetch("http://localhost:5000/download_final_csv", {
      method: "GET"
    });
    if(response.ok){
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "最终筛选特征值.csv");
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    }else{
      ElMessage.error('发生错误')
      return;
    }
  } 
  catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txt");
  }
}

const downloadTxt = async () => {
  try {
    const response = await fetch("http://localhost:5000/download_txt", {
      method: "GET"
    });
    if(response.ok){
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "原始公式.txt");
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    }else{
      ElMessage.error('发生错误')
      return;
    }
  } 
  catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txt");
  }
};

const downloadtxtNames = async () => {
  try {
    const response = await fetch("http://localhost:5000/download_txt_names", {
      method: "GET"
    });
    if (response.ok) {
      const data = await response.blob();
      const url = window.URL.createObjectURL(data);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute(
        "download",
        "特征公式.txt"
      );
      document.body.appendChild(link);
      link.click();
      ElMessage.success("下载成功");
    } else {
      ElMessage.error("Failed to download model txtNames: " + response.statusText);
    }
  } catch (error) {
    console.error("Error:", error);
    ElMessage.error("Failed to download model txtNames");
  }
};

const forthStep = async () => {
  const formData = new FormData();
  formData.append("file", files.value[0]);
  formData.append("best_params", JSON.stringify(form.value)); 
  fetch("http://localhost:5000/get_gp_features", {
    method: "POST",
    body: formData // 将前端获取的 form 传给后端
  }).then(response => {
    if (!response.ok) {
      ElMessage.error('发生错误')
      throw new Error("网络响应不正常");
    }
    return response.json();
  });
};

const nextStep = () => {
  if (parseInt(nSearches.value) <= 0) {
    ElMessage.error("请输入大于0的整数");
    return;
  }
  (async () => {
    await nextTick();
    if (nSearches.value.trim() === "") {
      ElMessage.error("请输入搜索次数。");
      return;
    }
    if (!/^\d+$/.test(nSearches.value)) {
      ElMessage.error("搜索次数必须为整数。");
      return;
    }
    activeStep.value++;
  })();
};

const uploadRemove = ()=>{
  files.value=[]
}

const prevStep = () => {
  // files.value[0] == null;
  activeStep.value--;
};

const beforeUpload = (file: File) => {
  if (file.name.split(".")[1] !== "xlsx") {
    ElMessage.error("请上传.xlsx文件");
    return false;
  }
};

const uploadSuccess = (res, file, filelist) => {
  console.log(res, "resrsrsersr");
  files.value.push(file.raw); // 将上传成功的文件添加到 files 中
  ElMessage({ type: "success", message: "上传成功" });
};

const stepThird = () => {
  const formData = new FormData();
  formData.append("file", files.value[0]);
  formData.append("best_params", JSON.stringify(form.value)); 
  vloading.value = true
  fetch("http://localhost:5000/generate_features", {
    method: "POST",
    body: formData, // 将前端获取的 form 传给后端
  })
    .then(response => {
      if (!response.ok) {
        throw new Error("网络响应不正常");
      }
      activeStep.value = 3
      return response.json();
    })
    .then(data => {
      gpFeatures.value = data
      console.log(gpFeatures.value);
    })
    .catch(error => {
      ElMessage.error('error:',error)
      vloading.value = false
    })
    .finally(() => {
      csvLoading.value = false;
      vloading.value = false
    });
};

const submit = () => {
  if (!files.value || files.value.length === 0) {
    ElMessage.error("请上传文件。");
    return;
  }
  const formData = new FormData();
  formData.append("file", files.value[0]);
  formData.append("n_searches", nSearches.value);
  formData.append('cv',cv.value);
  formData.append(
    "population_size",
    `${populationSizeMin.value},${populationSizeMax.value}`
  );
  formData.append(
    "generations",
    `${generationsMin.value},${generationsMax.value}`
  );
  formData.append("maxSamples", maxSamples.value);
  formData.append(
    "pCrossover",
    `${pCrossoverMin.value},${pCrossoverMax.value}`
  );
  formData.append(
    "pSubtreeMutation",
    `${pSubtreeMutationMin.value},${pSubtreeMutationMax.value}`
  );
  formData.append(
    "pHoistMutation",
    `${pHoistMutationMin.value},${pHoistMutationMax.value}`
  );
  formData.append(
    "pPointMutation",
    `${pPointMutationMin.value},${pPointMutationMax.value}`
  );
  formData.append(
    "parsimonyCoefficient",
    `${parsimonycoefficientMin.value},${parsimonycoefficientMax.value}`
  );
  formData.append(
    "tournamentSize", 
  `${tournamentMin.value},${tournamentMax.value}`
  )
  formData.append(
    "randomState", 
  `${randomstateMin.value},${randomstateMax.value}`
  )
  formData.append("functionSet", functionSet.value);
  subLoading.value = true;
  vloading.value = true;
  fetch("http://localhost:5000/file1", {
    method: "POST",
    body: formData
  })
    .then(response => {
      if (!response.ok) {
        throw new Error("网络响应不正常");
      }
      return response.json();
    })
    .then(data => {
      handleBackendResponse(data);
      // forthStep();
      activeStep.value = 2;
    })
    .catch(error => {
      console.error("发生了问题：", error);
    })
    .finally(() => {
      subLoading.value = false;
      vloading.value = false;
    });
};
const handleError = (error, file, fileList) => {
  ElMessage.error('上传文件发生错误，请检查文件是否正确');
};

</script>


<style>
.el-form-item__label {
	font-weight: 500;
}
.el-input__increase{
  display: none;
}
input[type='number']{
  -moz-appearance: textfield;
}
</style>