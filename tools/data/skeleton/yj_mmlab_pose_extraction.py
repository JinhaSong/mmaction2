#폴더 안에 있는 모든 클립 파일들에 대해서 포즈를 추출해서 지정한 경로로 ".pkl"파일을 만들어주는 코드
import subprocess

subprocess.call("ntu_pose_extraction.py",shell=True)