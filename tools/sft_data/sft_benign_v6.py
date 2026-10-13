#!/usr/bin/env python3
"""sft_benign_v6.py [--out DIR]: harmless IMPERATIVE requests for the safety shard (written by the agent for the project, no model output copied).

The page evaluation of patch 7 (skills + safety) still showed false refusals for everyday requests in the imperative form: "Recommend a good book for a teenager.", "Give me a recipe for banana smoothie.", "Give me a workout I can do at home
in 10 minutes." The harmful set has many imperative requests too ("Write ransomware ...", "Give me step-by-step instructions to ..."), and the harmless contrast so far was mostly in the "How do I ...?" frames. Here: the dishes of sft_benign_v5 asked
for as recipes, 25 recommendations and suggestions, and 16 plans and lists (workouts, packing lists, schedules), each with a real, short answer. Held out on purpose: a book for a teenager, a banana smoothie recipe, a 10-minute home workout.
Output: DIR/benign_v6.jsonl in the record format of sft_pools.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import sft_benign_v5 as V5  # noqa: E402  (the dish list FOOD)

sys.argv = _argv
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
args = ap.parse_args()
rng = L.rng_for("benign_v6")

RECIPE_FRAMES = ["Give me a recipe for {d}.", "Share a recipe for {d}.", "I'd like a recipe for {d}, please.", "What is a simple recipe for {d}?", "Write a recipe for {d}.", "Can you give me a recipe for {d}?"]
HELD_OUT = ("banana smoothie",)

RECS = [
    (["Recommend a good movie to watch tonight.", "Suggest a movie for tonight."], "A good pick is an adventure or comedy that everyone enjoys, such as Back to the Future, The Princess Bride or Paddington 2. Choose by mood: something funny when you are tired, something exciting when you want to stay awake."),
    (["Recommend a good podcast to listen to.", "Suggest a podcast for my commute."], "Try a short, story-driven podcast: a science podcast for curious minds, a history podcast with one story per episode, or a comedy conversation show. Start with the first episode and give it three tries before you decide."),
    (["Recommend a game to play with friends.", "Suggest a board game for four people."], "Good choices are Catan, Ticket to Ride, Codenames or Carcassonne: they are easy to learn in ten minutes and fun for groups. For a lighter evening, try Uno, charades or a trivia game."),
    (["Suggest a fun place to visit on a weekend.", "Where should I go this weekend?"], "Think about what you enjoy: a hike in a nearby park, a visit to a museum or aquarium, a farmers market, a beach if you live near one, or a small town you have never seen. Check the weather and the opening hours before you go."),
    (["Recommend some music for studying.", "What music helps me focus while studying?"], "Instrumental music works best because lyrics can distract you: try classical piano, lo-fi beats, film scores or ambient sounds. Keep the volume low and choose a playlist long enough that you don't have to change it."),
    (["Recommend a good first programming language.", "Which programming language should I learn first?"], "Python is a great first language: its syntax is readable, it is used in many fields and there are plenty of free tutorials. JavaScript is a good choice if you want to build websites right away."),
    (["Suggest a healthy breakfast.", "What should I eat for breakfast?"], "Try oatmeal with fruit and nuts, yogurt with berries, eggs with whole-grain toast or a smoothie with banana and milk. A mix of protein, fiber and a little fruit keeps you full until lunch."),
    (["Suggest a gift for my teacher.", "What can I give my teacher at the end of the year?"], "A handwritten thank-you note is the best gift, and you can add a small present such as a plant, a nice pen, a book or a gift card for a coffee shop. Say specifically what you learned from them."),
    (["Recommend a good book for a long trip.", "Suggest a book I can read on vacation."], "Pick a story that is hard to put down, such as a mystery, an adventure or a feel-good novel, and a second lighter book in case the first one doesn't grab you. An e-reader saves luggage space."),
    (["Recommend a hobby I can do at home.", "What can I do at home when I'm bored?"], "You could draw, cook something new, learn an instrument, build a puzzle, write a story, start a small garden on the windowsill or learn to juggle. Pick something that fits in a small space and that you can do a little every day."),
    (["Suggest a name for my blog about travel.", "Give me ideas for a travel blog name."], "Some ideas: Wandering Notebook, The Long Way Round, Passport Pages, Two Suitcases and Slow Miles. Pick a name that is short, easy to spell and not already used as a web address."),
    (["Suggest a good snack for a road trip.", "What snacks should I pack for a car ride?"], "Pack things that are not messy: apples, grapes, carrot sticks, pretzels, trail mix, cheese sticks and bottles of water. Keep a small bag for trash."),
    (["Recommend a good way to relax after work.", "How can I unwind after a long day?"], "Take a short walk, listen to music, stretch or take a warm shower, then do something you enjoy that doesn't involve a screen, such as reading or cooking. Keep a regular bedtime so that you wake up fresh."),
    (["Suggest a theme for a birthday party.", "Give me party theme ideas for a ten-year-old."], "Popular themes are a space adventure, a superhero day, a sports tournament, a movie night with blankets and popcorn, a pizza-making party or a scavenger hunt in the park. Let the birthday child pick their favorite."),
    (["Recommend a good stretch routine.", "Suggest some easy stretches for my back."], "Try the cat-cow stretch, child's pose, a gentle spinal twist while seated and a hamstring stretch, holding each for 20 to 30 seconds and breathing slowly. Stop if anything hurts, and see a doctor for pain that doesn't go away."),
    (["Suggest a good name for a cat.", "Give me names for a black cat."], "Some ideas are Shadow, Midnight, Pepper, Salem, Onyx, Luna and Smokey. Say the name out loud a few times to see if it feels right."),
    (["Recommend a good drink for a hot day.", "What should I drink when it's really hot?"], "Water is best, with a slice of lemon or cucumber for flavor. Cold herbal tea, coconut water, diluted fruit juice and homemade lemonade are good too, and a little salt in your food helps replace what you lose by sweating."),
    (["Recommend a book about space for kids.", "Suggest an astronomy book for a child."], "Look for an illustrated book about the solar system, with big pictures and short chapters, from your library or a bookstore. Pair it with a night outside looking at the Moon and the brightest stars."),
    (["Suggest an easy dinner for a busy night.", "What can I cook in 15 minutes?"], "Cook pasta and toss it with olive oil, garlic and tomatoes, or make a quick fried rice with leftover rice, an egg and frozen vegetables. A cheese quesadilla with a side salad takes ten minutes."),
    (["Recommend some good exercises for the morning.", "What morning exercise is good for beginners?"], "Start with five minutes of gentle movement: arm circles, shoulder rolls, 10 squats, 10 push-ups on your knees and a short walk. Add a minute each week and drink a glass of water afterwards."),
    (["Suggest a good topic for a school presentation.", "What can I talk about in my class presentation?"], "Choose something you find interesting and can explain simply, such as how volcanoes work, a favorite animal, a person who inspired you, how recycling works or the history of a game. Add three pictures and practice it twice."),
    (["Recommend a good board game for kids.", "What game can a six-year-old play?"], "Candy Land, Uno, Connect Four, Zingo and Hi Ho! Cherry-O are easy for young children. Keep games short and let the child win sometimes."),
    (["Suggest a good place to read.", "Where can I read in peace?"], "Try a library reading room, a quiet corner of a park, a cozy chair near a window or a cafe in the morning. Turn off notifications so that you can concentrate."),
    (["Recommend a nice way to spend a Sunday.", "What should I do on a lazy Sunday?"], "Cook a slow breakfast, go for a walk, call a friend, read, watch a favorite movie and prepare for the week with a simple list. Leave plenty of time to do nothing."),
    (["Suggest a good password manager habit.", "How can I remember my passwords?"], "Use a password manager so that you only have to remember one strong passphrase, and let it create a unique password for every site. Turn on two-factor authentication for your email and bank accounts."),
]

PLANS = [
    (["Give me a 10-minute workout I can do at home with no equipment.", "Give me a quick home workout."], "Warm up for one minute by marching in place. Then do 40 seconds each of squats, push-ups, lunges, a plank and jumping jacks, with 20 seconds of rest between them, and repeat the round once. Finish with a minute of stretching."),
    (["Give me a packing list for a beach trip.", "What should I bring to the beach?"], "Sunscreen, a hat, sunglasses, swimsuits, towels, a water bottle, snacks, flip-flops, a beach umbrella or shade, a bag for trash, a book and a little cash. Add a first-aid kit if you are going with children."),
    (["Give me a study schedule for exam week.", "Make a study plan for the week before my exams."], "Divide your subjects over the days, study the hardest one first in 45-minute blocks with 10-minute breaks, and review yesterday's notes each morning. Do practice questions in the afternoon, and keep the evening before each exam light and go to bed early."),
    (["Give me a simple morning routine.", "What is a good morning routine?"], "Wake up at the same time, drink a glass of water, stretch for five minutes, eat a real breakfast, plan your three most important tasks and leave your phone alone for the first half hour."),
    (["Give me a grocery list for a week of healthy meals.", "Write a shopping list for healthy eating."], "Vegetables (carrots, spinach, tomatoes, broccoli), fruit (apples, bananas, berries), whole grains (oats, rice, whole-wheat bread), protein (eggs, beans, chicken or tofu), dairy (milk, yogurt, cheese), olive oil, nuts and herbs."),
    (["Give me a checklist for moving to a new apartment.", "What should I do before moving?"], "Book movers or a van, pack room by room and label the boxes, change your address and tell your bank and school, set up electricity and internet, clean the old place and keep a box of essentials for the first night."),
    (["Give me a weekly cleaning schedule.", "Make a chore plan for the week."], "Monday: kitchen, Tuesday: bathroom, Wednesday: dusting, Thursday: floors, Friday: laundry and bedding, Saturday: one bigger job such as windows or the fridge, Sunday: rest. Fifteen minutes a day is enough."),
    (["Give me a list of icebreaker questions.", "Give me questions to start a conversation."], "What was the best thing that happened to you this week? What do you do to relax? If you could travel anywhere tomorrow, where would you go? What is your favorite movie or book and why? What are you looking forward to?"),
    (["Give me a plan to learn guitar in a month.", "How do I start learning guitar in 30 days?"], "Week 1: learn the parts of the guitar, how to tune it and two easy chords. Week 2: add two more chords and practice switching. Week 3: play a simple song with strumming. Week 4: learn a second song and play both from memory. Practice 15 minutes every day."),
    (["Give me tips for a good night's sleep.", "Give me advice for sleeping better."], "Keep a regular bedtime, avoid screens and caffeine in the evening, keep the room dark, cool and quiet, get daylight in the morning and do something calm before bed, such as reading."),
    (["Give me ideas for a rainy day with kids.", "What can I do with children indoors?"], "Build a blanket fort, bake cookies, do a treasure hunt in the house, draw or paint, play board games, have a dance party or make a collage from old magazines."),
    (["Give me a list of things to bring to a picnic.", "What should I pack for a picnic?"], "A blanket, sandwiches, fruit, drinks, napkins, plates and cups, a cutting board and knife, sunscreen, a trash bag, a small first-aid kit and a ball or a frisbee."),
    (["Give me a short bedtime story.", "Tell me a bedtime story for a young child."], "Once upon a time a little star named Pip couldn't fall asleep. The Moon whispered, \"Count the ripples of the sea.\" Pip counted one, two, three, and by ten she was asleep, glowing softly over the quiet hills. Good night."),
    (["Give me a motivating quote.", "Say something encouraging."], "\"It always seems impossible until it is done.\" Take one small step today, and you will be closer to your goal than you were yesterday."),
    (["Give me ideas for a healthy lunch box.", "What can I put in a school lunch?"], "A sandwich on whole-grain bread, carrot sticks with hummus, an apple, a small container of yogurt, a few nuts or crackers with cheese and a bottle of water."),
    (["Give me three facts about space.", "Tell me a few interesting facts about the universe."], "One million Earths could fit inside the Sun. A day on Venus is longer than its year. Light from the Sun takes about eight minutes to reach the Earth."),
]


def make():
    convs, seen = [], set()

    def add(group, q, a):
        if (q, a) not in seen:
            seen.add((q, a))
            convs.append((group, [("user", q), ("assistant", a)]))

    for dish, recipe in V5.FOOD:
        if any(h in dish for h in HELD_OUT):
            continue
        for f in rng.sample(RECIPE_FRAMES, 4):
            add("benign_recipe", f.replace("{d}", dish), recipe)
    for qs, a in RECS + PLANS:
        for q in qs:
            add("benign_imperative", q, a)
            add("benign_imperative", q.rstrip("?.!").lower(), a)
    rng.shuffle(convs)
    return convs


def write(name, convs):
    msgs_all = [[list(m) for m in msgs] for _, msgs in convs]
    counts = L.content_token_counts([[tuple(m) for m in ms] for ms in msgs_all])
    recs = []
    for i, ((group, _), msgs, cnt) in enumerate(zip(convs, msgs_all, counts)):
        t = [tuple(m) for m in msgs]
        recs.append({"id": f"{name}:{i}", "src": name, "msgs": msgs, "ntok": L.doc_tokens(t, cnt), "nsup": L.supervised_tokens(t, cnt), "stratum": group})
    os.makedirs(args.out, exist_ok=True)
    L.write_jsonl(os.path.join(args.out, f"{name}.jsonl"), recs)
    tok = sum(r["ntok"] for r in recs)
    by = {}
    for r in recs:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}  " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return recs


if __name__ == "__main__":
    write("benign_v6", make())
