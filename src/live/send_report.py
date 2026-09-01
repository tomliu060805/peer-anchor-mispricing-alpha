# -*- coding: utf-8 -*-
"""把当周 Top-50 报告发到邮箱。

凭证从环境变量读取, 绝不写进代码或仓库:
    SMTP_HOST   默认 smtp.qq.com
    SMTP_PORT   默认 465 (SSL)
    SMTP_USER   发信邮箱
    SMTP_PASS   **授权码**, 不是登录密码 (QQ邮箱: 设置->账号->开启SMTP服务后获取)
    MAIL_TO     收件人, 多个用逗号分隔

放在项目根的 .env 里(已在 .gitignore 中)。缺任一项则跳过发信并明确报错,
不静默失败——静默失败的定时任务等于没有。

用法:
    python src/live/send_report.py                  # 发最新一期
    python src/live/send_report.py --asof 2026-08-31
    python src/live/send_report.py --test           # 只发一封连通性测试信
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from paths import PROJ_ROOT as PROJ, OUTPUT_DIR

import os, ssl, glob, smtplib, argparse
from email.message import EmailMessage


def _cfg():
    miss = [k for k in ('SMTP_USER', 'SMTP_PASS', 'MAIL_TO') if not os.environ.get(k)]
    if miss:
        raise SystemExit(f'缺少环境变量 {miss}——请在项目根 .env 中配置后重试。\n'
                         f'注意 SMTP_PASS 填的是邮箱**授权码**, 不是登录密码。')
    return {'host': os.environ.get('SMTP_HOST', 'smtp.qq.com'),
            'port': int(os.environ.get('SMTP_PORT', '465')),
            'user': os.environ['SMTP_USER'], 'pw': os.environ['SMTP_PASS'],
            'to': [x.strip() for x in os.environ['MAIL_TO'].split(',') if x.strip()]}


def send(subject, body, attachments=(), c=None):
    c = c or _cfg()
    m = EmailMessage()
    m['Subject'] = subject
    m['From'] = c['user']
    m['To'] = ', '.join(c['to'])
    m.set_content(body)
    for p in attachments:
        if not os.path.exists(p):
            continue
        with open(p, 'rb') as f:
            m.add_attachment(f.read(), maintype='text', subtype='csv',
                             filename=os.path.basename(p))
    ctx = ssl.create_default_context()
    if c['port'] == 465:
        with smtplib.SMTP_SSL(c['host'], c['port'], context=ctx, timeout=30) as s:
            s.login(c['user'], c['pw'])
            s.send_message(m)
    else:
        with smtplib.SMTP(c['host'], c['port'], timeout=30) as s:
            s.starttls(context=ctx)
            s.login(c['user'], c['pw'])
            s.send_message(m)
    print(f'已发送至 {m["To"]}: {subject}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asof', default=None)
    ap.add_argument('--top', type=int, default=50)
    ap.add_argument('--test', action='store_true')
    a = ap.parse_args()
    c = _cfg()
    if a.test:
        send('[联动锚定策略] SMTP 连通性测试',
             '这是一封测试信。收到即说明定时任务可以把每周 Top-50 发到这里。', c=c)
        return
    live = f'{OUTPUT_DIR}/live'
    if a.asof:
        txt = f'{live}/top{a.top}_{a.asof}.txt'
    else:
        cand = sorted(glob.glob(f'{live}/top{a.top}_*.txt'))
        if not cand:
            raise SystemExit(f'{live} 下没有 top{a.top}_*.txt, 请先跑 weekly_top50.py')
        txt = cand[-1]
    ds = os.path.basename(txt).replace(f'top{a.top}_', '').replace('.txt', '')
    body = open(txt, encoding='utf-8').read()
    atts = [f'{live}/top{a.top}_{ds}.csv', f'{live}/portfolio_{ds}.csv',
            f'{live}/orders_{ds}.csv']
    send(f'[联动锚定策略] {ds} 打分 Top-{a.top}', body, atts, c=c)


if __name__ == '__main__':
    main()
