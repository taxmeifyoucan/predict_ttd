<template>
  <div>
    <h2> The Merge TTD prediction</h2>
    <p><a href=https://github.com/taxmeifyoucan/predict_ttd/>Make your own prediction and contribute</a></p>
    <p>Total Terminal Difficulty of {{ target }}is expected around {{ msg }}</p>
    <img src="../../../api/chart.png" alt="">
    <img src="../../../api/hashrate.png" alt="">

  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'Update',
  data() {
    return {
      msg: '',
      target: '',
    };
  },
  methods: {
    getPredict() {
      axios.get('/ttd_prediction')
        .then((res) => {
          this.msg = res.data;
        })
        .catch((error) => {
          console.error(error);
        });
    },
    getTarget() {
      axios.get('/ttd_target')
        .then((res) => {
          this.target = res.data;
        })
        .catch((error) => {
          console.error(error);
        });
    },
  },
  created() {
    this.getPredict();
    this.getTarget();

  },
};
</script>

