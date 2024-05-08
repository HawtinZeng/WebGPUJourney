import redCom from "./reduce.wgsl.js";
const adapter = await navigator.gpu?.requestAdapter();
const device = await adapter?.requestDevice();
if (!device) {
  fail("need a browser that supports WebGPU");
}
const module = device.createShaderModule({
  label: "reducer",
  code: `
  @group(0) @binding(0) var<storage,read> inputBuffer: array<u32>;
  @group(0) @binding(1) var<storage,read_write> output: atomic<u32>;
  
  // Create zero-initialized workgroup shared data
  const wgsize : u32 = 256;
  var<workgroup> sdata: array<u32, wgsize>;
  
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
  
      // Each thread should read its data:
      var tid: u32 = local_id.x;
      sdata[tid] = select(0, inputBuffer[id.x], id.x < 4194304);
  
      // sync all the threads:
      workgroupBarrier();
  
      // Do the reduction in shared memory: reduction
      for (var s: u32 = 1; s < wgsize; s *= 2) {
          if tid % (2 * s) == 0 {
              sdata[tid] += sdata[tid + s];
          }
  
          workgroupBarrier();
      }
  
      // Add result from the workgroup to the output storage:
      if tid == 0 {
          atomicAdd(&output, sdata[0]);
      }
  }`,
});

const pipeline = device.createComputePipeline({
  label: "sum",
  layout: "auto",
  compute: {
    module,
  },
});

const outputSize = 4;
const outputStorageBuffer = device.createBuffer({
  label: "storage",
  size: outputSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
// const outputValues = new Int32Array(1);

const sumValues = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
const sumValuesBuffer = device.createBuffer({
  label: "sum values",
  size: sumValues.length * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(sumValuesBuffer, 0, sumValues);

const bindGroup = device.createBindGroup({
  label: "sum bind group",
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    {
      binding: 0,
      resource: {
        buffer: sumValuesBuffer,
      },
    },
    {
      binding: 1,
      resource: {
        buffer: outputStorageBuffer,
      },
    },
  ],
});

const encoder = device.createCommandEncoder();
const computePass = encoder.beginComputePass();

computePass.setBindGroup(0, bindGroup);
computePass.setPipeline(pipeline);

computePass.dispatchWorkgroups(16, 1);
computePass.end();
