# 動作確認
test:
	python3 app.py test

# ラベリングで処理
use-lageling:
	python3 app.py test labeling

# 輪郭データで処理
use-cont:
	python3 app.py test cont

# # ラベリングのテスト
# labeling-test:
# 	python3 app.py labeling-test


# 本番動作
exe:
	python3 app.py all

