{{/*
Expand the name of the chart.
*/}}
{{- define "ml-serving.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "ml-serving.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label.
*/}}
{{- define "ml-serving.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ml-serving.labels" -}}
helm.sh/chart: {{ include "ml-serving.chart" . }}
{{ include "ml-serving.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ad-ml-platform
meta.fb.com/team: ml-platform
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ml-serving.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-serving.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: ml-serving
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "ml-serving.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-serving.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
GPU tolerations
*/}}
{{- define "ml-serving.gpuTolerations" -}}
{{- if .Values.gpu.enabled }}
{{- toYaml .Values.gpu.tolerations }}
{{- end }}
{{- end }}

{{/*
GPU nodeSelector
*/}}
{{- define "ml-serving.gpuNodeSelector" -}}
{{- if .Values.gpu.enabled }}
{{- toYaml .Values.gpu.nodeSelector }}
{{- end }}
{{- end }}

{{/*
Image reference
*/}}
{{- define "ml-serving.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}
