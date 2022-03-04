import numpy as np
import pandas as pd
import time
import base64
import pytesseract
import sys
import datetime
import termcolor
import requests
import pandas_ta as ta

from numpy.random import random

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def writefile(name_file, line):
    f = open(name_file, "a+")
    for a in range(len(line)):
        f.write('%02d,%02d,%02d,%02d,%.9f'%(line[a][0], line[a][1], line[a][2], line[a][3], line[a][4])+"\n")
    f.close()

def printColor(a, b, p, money, changes, entries, data):
    if len(data) > 0:
        v = (a+b)/2
        c = np.sign(v-data[-1][-1])
        color = "on_green" if c > 0 else ("on_red" if c < 0 else "on_grey")
        termcolor.cprint("%f %f %d"%(a, b, p), "white", color, end="")
        text_money = "%.2f"%(money) if money > 0 else "Money error!"
        print("  ",end="")
        termcolor.cprint(text_money, "white", "on_yellow", end="")
        acc = float(changes-entries)/entries if entries > 0 else -1
        print("  ",end="")
        termcolor.cprint("%.2f"%(acc), "white", "on_magenta")

def money_log(money, direction, payout):
    with open("money_log.csv", "a+") as f:
        f.write("%.2f,%d,%d\n"%(money, direction, payout))

def open_position(driver, direction):
    try:
        if direction > 0:
            driver.find_element_by_xpath("//button[@data-test='deal-button-up']").click()
        elif direction < 0:
            driver.find_element_by_xpath("//button[@data-test='deal-button-down']").click()
    except:
        print("W: buttons not found")

def refresh(driver):
    try:
        el = driver.find_element_by_xpath("//button[@data-test='deal-button-up']")
        action = webdriver.common.action_chains.ActionChains(driver)
        action.move_to_element_with_offset(el, -500, 0)
        action.click()
        action.move_to_element_with_offset(el, -200, 0)
        action.click()
        action.perform()
        print("refresh done")
    except:
        print(("W: refresh failure"))

def enable_sentiment(driver):
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//button[@data-test='indicator-menu']")))
    driver.find_element_by_xpath("//button[@data-test='indicator-menu']").click()
    while True:
        try:
            driver.find_element_by_xpath("//span[@data-trans='indicators_menu_title_sentiment']").click()
            driver.find_element_by_xpath("//button[@data-test='cor-w-panel-close']").click()
            break
        except Exception as e:
            continue

def select_best_asset(driver):
    try:
        driver.find_element_by_class_name("ButtonBase-module-host-3Fo.PushNotificationContent-module-close-3Oc").click()
        time.sleep(1)
    except Exception as e:
        print(str(e))
    
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//button[@data-test='asset-select-button']")))
    driver.find_element_by_xpath("//button[@data-test='asset-select-button']").click()
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//div[@data-test='asset-tags-item']")))
    el = driver.find_elements_by_xpath("//div[@data-test='asset-tags-item']")
    time.sleep(0.5)
    for element in el:
        try:
            element.click()
            WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.XPATH, "//div[@data-test='asset-item-title']")))
            driver.find_elements_by_xpath("//div[@data-test='asset-item-title']")[0].click()
            break
        except Exception as e:
            print(str(e))
    try:
        driver.find_element_by_xpath("//button[@data-test='cor-w-panel-close']").click()
    except:
        pass

def rsi(data, n = 60):
    df = pd.DataFrame(data)
    data_rsi = ta.rsi(df[0], n).to_numpy()
    data_rsi[np.isnan(data_rsi)] = 50
    data_rsi = (data_rsi/50)-1
    return data_rsi[-1]

class Model():
    def __init__(self, driver, pred_time = 60, lost_cooldown = 120):
        self.driver = driver
        self.data = []
        self.s = []
        self.cooldown = 0
        self.pred_time = pred_time
        self.lost_cooldown = lost_cooldown
        self.money = 100.0
        self.to_close = 0
        self.dir_position = 0
        self.open_value = 0
        self.lim_dn = -1
        self.lim_up = 1
        self.refresh = 1*60*60
        self.alpha = 2
        select_best_asset(self.driver)

    def iterate(self, value, payout, money, sentiment):
        self.data.append(value)
        self.s.append(sentiment)

        if len(self.data) > 1:
            if np.abs(self.data[-1]-self.data[-2]) > 0.01*self.data[-2]:
                self.data = []
                self.s = []
                print("asset changed, all data discarded...")
                return

        def simulate_alpha(data, alpha, n = 14, timeout = 5, payout = 0.8):
            money = 0
            cooldown_trade = 0
            for i in range(len(data)-timeout):
                value = data[i]
                if i >= n:
                    cooldown_trade -= 1
                    if cooldown_trade < 0:
                        avg = np.sum(data[i-n:i])/n
                        std = np.std(data[i-n:i])
                        ub = avg + alpha*std
                        lb = avg - alpha*std
                        if value > ub or value < lb:
                            result = np.sign(-(value-avg))*np.sign(data[i+timeout]-value)
                            if result > 0:
                                money += payout
                            elif result < 0:
                                money -= 1
                            cooldown_trade = timeout
            return money

        def adjust_interval(data, t_max = 240):
            data_local = data[-t_max:] if len(data) > t_max else data
            alpha_max = np.inf
            score_max = -np.inf
            for alpha in np.linspace(0.5, 4.5, 16+1):
                score = simulate_alpha(data_local, alpha)
                if score > score_max:
                    alpha_max = alpha
                    score_max = score
            print(alpha_max, score_max)
            return alpha_max

        if len(self.data) % 60 == 0:
            self.alpha = adjust_interval(self.data)
            print(self.alpha)

        if len(self.s) > 14:
            if np.abs(self.data[-1]-np.sum(self.data[-14:])/14) > self.alpha*np.std(self.data[-14:]):
                open_position(self.driver, -np.sign(self.data[-1]-np.sum(self.data[-14:])/14))

        if payout < 80:
            select_best_asset(self.driver)
        
        print(value, payout, money, sentiment)

if len(sys.argv) > 2:
    driver = webdriver.Chrome(ChromeDriverManager().install())
else:
    driver = webdriver.Firefox()
driver.set_window_size(1366, 768)
print("loading login page...")
driver.get("https://olymptrade.com/")

WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//button[@data-test='action-sign-in']")))
driver.find_element_by_xpath("//button[@data-test='action-sign-in']").click()

WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.NAME, "email")))

username = driver.find_element_by_name("email")
password = driver.find_element_by_name("password")

with open("passwd","r") as f:
    for line in f:
        linestrs = line.split(" ")
        username.send_keys(linestrs[0])
        password.send_keys(linestrs[1])

input("please solve captcha and press enter...")
driver.find_element_by_xpath("//button[@type='submit']").click()
print("login sent...")

WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CLASS_NAME, "pin_text")))
time.sleep(2)

enable_sentiment(driver)

refresh_cooldown = 60

filename = sys.argv[1] if len(sys.argv) > 1 else "temp.sr"
time_predict = 60

money = -1
payout = -1

model = Model(driver, 240, 660)
while True:
    try:
        elements = driver.find_elements_by_class_name("pin_text")
        pins = []
        for el in elements:
            pins.append(float(el.get_attribute("innerHTML")))
        value = np.median(pins)
        money_str = driver.find_element_by_xpath("//div[@data-test='account-balance-value']").find_element_by_xpath(".//span[@dir='ltr']").get_attribute("innerHTML")
        money = float(''.join(filter(str.isdigit, money_str)))/100
        payout_str = driver.find_element_by_xpath("//span[@data-test='asset-item-bage']").find_element_by_xpath(".//span[@class='']").get_attribute("innerHTML")
        payout = float(''.join(filter(str.isdigit, payout_str)))
        try:
            sentiment_str = driver.find_element_by_class_name("sentiment--text.sentiment--text__down").get_attribute("innerHTML")#("//div[@class='sentiment--text__down']").get_attribute('innerHTML')
            sentiment = (float(''.join(filter(str.isdigit, sentiment_str)))-50)*0.02
        except Exception as e:
            print(str(e))
            sentiment = 0
    except:
        print("error, retrying...")
        continue
    
    model.iterate(value, payout, money, sentiment) # must respond in last than 1sec

    with open(filename, "a") as f:
        f.write("%f,%f,%f\n"%(value, payout, sentiment))
    
    refresh_cooldown -= 1
    if refresh_cooldown <= 0:
        refresh(driver)
        refresh_cooldown = 60

    time.sleep((1000-round(datetime.datetime.now().microsecond/1000.0))/1000.0) # wait next second