<template>
  <div>
    <h2> Ropsten TTD predictions </h2>
    <p><a href=https://github.com/taxmeifyoucan/predict_ttd/>Contribute and make your own prediction</a></p>
    <p>Total Terminal Difficulty of {{ target }}is expected around {{ msg }}</p>
    <img src="../assets/chart.png" alt="">

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
      axios.get('/prediction')
        .then((res) => {
          this.msg = res.data;
        })
        .catch((error) => {
          console.error(error);
        });
    },
    getTarget() {
      axios.get('/target')
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

