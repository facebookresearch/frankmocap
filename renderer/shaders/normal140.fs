//Copyright (c) Facebook, Inc. and its affiliates.

#version 140

out vec4 FragColor;
in vec3 CamNormal;

void main() 
{
    // FragColor = vec4(Color,1.0);
    vec3 cam_norm_normalized = normalize(CamNormal);
    vec3 rgb = (cam_norm_normalized + 1.0) / 2.0;
	FragColor = vec4(rgb, 1.0);
}