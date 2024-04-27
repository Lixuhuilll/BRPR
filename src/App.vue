<script setup lang="ts">
import { appWindow } from '@tauri-apps/api/window'
import { confirm } from '@tauri-apps/api/dialog'
import { computed, onMounted, reactive } from 'vue'
import NumberSelector from './components/NumberSelector.vue'

const bullets = reactive({
  real: parseInt(localStorage.getItem('real-bullets') ?? '0'),
  empty: parseInt(localStorage.getItem('empty-bullets') ?? '0')
})
const realProbability = computed(() =>
  Math.round((bullets.real / (bullets.real + bullets.empty)) * 100)
)

// i18n
const i18n = {
  title: 'Buckshot Roulette Projectile Recorder',
  confirmMsg: 'Confirm to quit?',
  okLabel: 'Ok',
  cancelLabel: 'Cancel',
  real: 'Real',
  empty: 'Empty',
  quantity: 'Q',
  probability: 'P'
}
if (navigator.language.toLocaleLowerCase().includes('zh-cn')) {
  i18n.title = '恶魔轮盘记弹器'
  i18n.confirmMsg = '确认退出？'
  i18n.okLabel = '确认'
  i18n.cancelLabel = '取消'
  i18n.real = '实弹'
  i18n.empty = '空包弹'
  i18n.quantity = '数量'
  i18n.probability = '概率'
}

onMounted(async () => {
  await appWindow.setTitle(i18n.title)
  await appWindow.setAlwaysOnTop(true)
  // 监听退出请求
  await appWindow.listen('tauri://close-requested', async () => {
    const result = await confirm(i18n.confirmMsg, {
      title: i18n.title,
      okLabel: i18n.okLabel,
      cancelLabel: i18n.cancelLabel,
      type: 'warning'
    })
    if (result) {
      // 退出前先保存数据
      localStorage.setItem('real-bullets', bullets.real.toString())
      localStorage.setItem('empty-bullets', bullets.empty.toString())
      await appWindow.close()
    }
  })
  // 在 tauri.conf.json 里配置窗口默认隐藏，等待页面准备好后再让窗口显示出来
  // 这样可以避免 tauri-plugin-window-state 导致窗口启动时闪动
  await appWindow.show()
})
</script>

<template>
  <div class="container">
    <div class="bullet-item">
      <span class="left">{{ i18n.real }}</span>
      <span class="right">{{ i18n.quantity }}：{{ bullets.real }}</span>
    </div>
    <div class="bullet-item">
      <NumberSelector class="left" color="darkred" :count="4" v-model="bullets.real" />
      <span class="right">{{ i18n.probability }}：{{ realProbability }}%</span>
    </div>
    <div class="divider"></div>
    <!-- 分割线 -->
    <div class="bullet-item">
      <span class="left">{{ i18n.empty }}</span>
      <span class="right">{{ i18n.quantity }}：{{ bullets.empty }}</span>
    </div>
    <div class="bullet-item">
      <NumberSelector class="left" color="darkgreen" :count="4" v-model="bullets.empty" />
      <span class="right">{{ i18n.probability }}：{{ 100 - realProbability }}%</span>
    </div>
  </div>
</template>

<style scoped>
.container {
  margin: 0;
  height: 95vh;
  padding-top: 5vh;
  padding-left: 3vw;
}

.bullet-item {
  display: flex;
  justify-content: space-between;
}

.divider {
  height: 0.5vh; /* 分割线的高度 */
  background-color: #ccc; /* 分割线的颜色，可以根据需要调整 */
  margin-top: 7vh;
  margin-bottom: 4vh;
  margin-right: 3vw;
}

.left {
  flex: 1.2;
}

.right {
  flex: 0.8;
}
</style>
