#!/usr/bin/env python3
"""sft_benign_v4.py [--out DIR]: more harmless look-alikes for the safety shard (written by the agent for the project, no model output copied).

After patch 3 the held-out probe still showed over-refusals on harmless requests that share a verb or a topic with the harmful ones (stop programs on Windows, volcano for a science fair, a villain's speech). This set widens the contrast with
real answers to: process and program management (kill, terminate, execute, uninstall), idioms with violent verbs (nail an interview, crush a fear, smash a serve, attack a to-do list), science and history questions about dangerous
things (reactors, lightning, gunpowder, the Manhattan Project, antibiotics), everyday safety, cooking and outdoor hobbies, and fiction with conflict (a thief, a duel with a dragon, a war poem).
Held out on purpose (probes A and B): background programs on Windows, a volcano for a science fair, a villain's speech, a debate, balloons, a Mac program that does not respond, fireworks colors, old batteries, a firewall, a webcam, a ghost story.
Output: DIR/benign_v4.jsonl in the record format of sft_pools.py (id, src, msgs, ntok, nsup, stratum).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
args = ap.parse_args()
rng = L.rng_for("benign_v4")

ITEMS = [
    # process and program management
    (["How do I uninstall a program on Windows?", "How can I remove an app I don't use from my Windows PC?", "uninstall software windows 11 how"],
     "Open Settings, go to Apps and then Installed apps, find the program, click the three dots and choose Uninstall. Follow the prompts, and restart the computer if it asks you to."),
    (["How do I stop a Docker container?", "What is the command to stop a running Docker container?", "docker stop how"],
     "List the running containers with `docker ps`, then run `docker stop CONTAINER_ID` (or the container's name). If it does not stop, use `docker kill CONTAINER_ID`."),
    (["How do I kill a process by name on Linux?", "How can I kill every process with a certain name in Linux?", "linux kill process by name"],
     "Use `pkill name` to send the process a polite stop signal, or `killall name`. If it refuses to quit, add `-9`, for example `pkill -9 name`, which ends it immediately."),
    (["How do I restart a service with systemctl?", "How can I stop and start a service on Ubuntu?", "systemctl restart service how"],
     "Run `sudo systemctl restart SERVICE` to restart it, or `sudo systemctl stop SERVICE` and `sudo systemctl start SERVICE` separately. Check the result with `systemctl status SERVICE`."),
    (["How do I clear the cache in Chrome?", "How can I delete my browsing data in Chrome?", "chrome clear cache"],
     "Press Ctrl+Shift+Delete (Cmd+Shift+Delete on a Mac), choose a time range, tick Cached images and files, and click Clear data."),
    (["How do I delete a Git branch?", "How can I remove a branch in git?", "git delete branch command"],
     "Delete a local branch with `git branch -d NAME` (use `-D` to force it). To delete it on the remote too, run `git push origin --delete NAME`."),
    (["How do I run a Python script from the terminal?", "How can I execute a .py file?", "run python file command line"],
     "Open a terminal, move to the folder with `cd`, and type `python script.py` (or `python3 script.py`). Make sure Python is installed by checking `python --version` first."),
    (["How do I free up disk space on Windows?", "My C: drive is almost full. What can I do?", "windows disk full what to delete"],
     "Run Disk Cleanup or Storage Sense to remove temporary files, empty the Recycle Bin, uninstall programs you no longer use and move large photos or videos to an external drive. The Storage page in Settings shows what takes the most room."),
    (["How do I stop a program from starting when I log in?", "How can I disable startup apps on Windows?", "disable startup programs"],
     "On Windows, press Ctrl+Shift+Esc, open the Startup apps tab, select the program and click Disable. On a Mac, open System Settings, then General, then Login Items, and remove it from the list."),
    (["How do I close all my browser tabs at once?", "What's the quickest way to close every tab in Firefox?", "close all tabs shortcut"],
     "Close the whole window with Ctrl+Shift+W (Cmd+Shift+W on a Mac), or right-click a tab and choose Close Other Tabs, then close the last one."),
    (["How do I make my computer faster?", "My laptop is slow. What can I try?", "speed up old pc tips"],
     "Restart it, close programs you aren't using, remove apps from startup and keep at least 15 percent of the disk free. Install updates, scan for malware and, on an old machine, adding memory or switching to an SSD helps the most."),
    (["How do I exit a terminal program that is stuck?", "How can I interrupt a command in the terminal?", "ctrl c stop command"],
     "Press Ctrl+C to interrupt the running command. If the program is frozen, open another terminal, find its process ID with `ps aux` and end it with `kill PID` (or `kill -9 PID` as a last resort)."),
    # idioms with violent verbs
    (["How do I nail a job interview?", "How can I crush an interview and get the job?", "tips to ace an interview"],
     "Research the company, practice answers to common questions out loud and prepare examples of your work. Arrive early, listen carefully, be specific about what you achieved and ask a couple of good questions at the end."),
    (["How do I beat writer's block?", "How can I kill writer's block when I can't start?", "stuck writing what to do"],
     "Lower the bar: write badly on purpose for ten minutes, or start with the part you know best instead of the beginning. Change the place, take a short walk, set a small word goal and edit later."),
    (["How do I smash my personal best in running?", "How can I beat my 5k time?", "improve running pace tips"],
     "Add one interval session a week, such as 6 x 400 meters at a hard pace with easy jogs in between, and keep the other runs easy. Sleep well, build up distance slowly and start the race at an even pace instead of sprinting."),
    (["How do I get through my to-do list when it feels too long?", "How can I finish a huge to-do list?", "overwhelmed by tasks help"],
     "Write everything down, pick the three most important tasks and do the quickest one first to build momentum. Block out time for each task, silence notifications and cross items off as you finish them."),
    (["How do I crush my fear of heights?", "How can I overcome a fear of flying?", "fear of public speaking how to beat"],
     "Face the fear in small steps: begin with a low balcony or a short flight, stay until the anxiety drops and then try a bit more. Slow breathing helps in the moment, and a counselor can guide you through gradual exposure."),
    (["How do I smash a tennis serve?", "How can I hit a harder serve in tennis?", "tennis serve power tips"],
     "Use a continental grip, toss the ball slightly in front of you and bend your knees. Swing up and through the ball, snap your wrist at contact and let your legs and hips add the power, not only the arm."),
    (["How do I blast through my reading list?", "How can I read more books this year?", "read more habit tips"],
     "Read a little every day, for example 20 pages or a chapter before bed, and always carry a book or an e-reader. Drop books that bore you, mix long and short ones and keep a list of what you want to read next."),
    (["How do I hit my savings goal faster?", "How can I save money quickly?", "ways to save money fast"],
     "Track your spending for a month, cut the biggest unnecessary costs, and move a fixed amount to savings on payday. Cook at home, cancel subscriptions you don't use and add a little income from a side job or by selling things you no longer need."),
    (["How do I kill the lag in my online game?", "How can I reduce lag when I play online?", "fix game lag ping"],
     "Use a wired connection instead of wifi, close downloads and streaming on other devices, and pick a server close to you. Lower the graphics settings and update your graphics drivers."),
    (["How do I steal the spotlight in a group project presentation?", "How can I make my part of the presentation stand out?", "memorable presentation tips"],
     "Open with a surprising fact or a short story, keep your slides simple and speak to the audience instead of reading. Practice your timing and end with one clear message people will remember."),
    # science and history
    (["How does a nuclear reactor work?", "Explain how nuclear power makes electricity.", "nuclear power plant simple explanation"],
     "In a reactor, atoms of uranium split and release heat in a controlled chain reaction. The heat turns water into steam, the steam spins a turbine, and a generator turns that motion into electricity. Control rods absorb neutrons to keep the reaction steady."),
    (["What causes lightning?", "How does lightning form?", "why do thunderstorms have lightning"],
     "Ice and water droplets rub against each other inside a storm cloud and separate electric charges. When the difference between the cloud and the ground, or between parts of the cloud, grows large enough, a huge spark jumps across: that spark is lightning, and the sudden heat makes the air boom as thunder."),
    (["What was the Manhattan Project?", "Tell me about the Manhattan Project.", "manhattan project summary"],
     "The Manhattan Project was a secret United States research program during World War II, with help from Britain and Canada, that developed the first atomic bombs. It started in 1942, employed more than 100,000 people and led to the first test in New Mexico in July 1945."),
    (["What is the history of gunpowder?", "Who invented gunpowder?", "where did gunpowder come from"],
     "Gunpowder was invented in China around the 9th century, probably by alchemists looking for an elixir of life. It was first used in fireworks and later in weapons, and it spread along trade routes to the Middle East and Europe by the 13th century."),
    (["How do airbags work in a car?", "What makes a car airbag inflate?", "airbag how does it work"],
     "Sensors detect a sudden, strong slowdown in a crash and send a signal that triggers a chemical reaction in the airbag module. The reaction produces gas in a fraction of a second, which fills the bag before the occupant moves forward, and the bag then deflates to absorb the impact."),
    (["How do antibiotics kill bacteria?", "What do antibiotics do?", "antibiotics explained simply"],
     "Antibiotics attack parts of bacteria that human cells don't have, such as their cell walls or the machinery they use to make proteins. They either kill the bacteria or stop them from multiplying, and they do not work against viruses."),
    (["What was the Battle of Hastings?", "Tell me about the Battle of Hastings.", "battle of hastings 1066 summary"],
     "The Battle of Hastings took place on 14 October 1066 in southern England. The Norman army of William the Conqueror defeated King Harold's English army, Harold was killed, and William became king, which began the Norman rule of England."),
    (["How did the Roman army train its soldiers?", "What was Roman military training like?", "roman legion training"],
     "Recruits learned to march long distances carrying heavy equipment, practiced fighting with wooden swords and shields, and drilled in formation until the movements were automatic. Discipline was strict, and the legions also built roads, camps and bridges."),
    (["What is the French Revolution?", "Explain the causes of the French Revolution.", "french revolution summary"],
     "The French Revolution began in 1789. Food shortages, heavy taxes on ordinary people, a government deep in debt and new ideas about equality and rights led the people to challenge the king. It ended the monarchy for a time and led to the Republic and later Napoleon."),
    (["Why do volcanoes have different kinds of eruptions?", "What decides whether an eruption is explosive?", "explosive vs effusive volcano"],
     "It depends on the magma. Thick, sticky magma with a lot of dissolved gas traps the gas until the pressure bursts out in an explosive eruption, while thin, runny magma lets the gas escape and flows out as lava."),
    (["How do fire extinguishers work?", "What is inside a fire extinguisher?", "extinguisher how to use PASS"],
     "An extinguisher sprays a substance, such as water, foam or a dry powder, that cools the fire, smothers it or interrupts the chemical reaction. To use one, remember PASS: Pull the pin, Aim at the base of the flames, Squeeze the handle and Sweep from side to side."),
    # everyday safety
    (["What should I do if I smell gas at home?", "I think there is a gas leak. What now?", "smell of gas in the kitchen what to do"],
     "Don't use switches, lighters or phones inside. Open windows if you can, leave the building right away and call the gas company or the emergency number from outside."),
    (["How do I childproof my home?", "How can I make my house safe for a toddler?", "baby proofing checklist"],
     "Cover electrical outlets, fit gates at stairs, anchor heavy furniture to the wall and lock cabinets that hold cleaning products and medicine. Keep small objects, cords and plastic bags out of reach."),
    (["How do I report a stolen bike?", "My bike was stolen. What should I do?", "bike theft what now"],
     "File a report with the police and give them the serial number, photos and where it was taken. Tell your insurance company, check local online marketplaces for it and register the serial number on a bike registry if there is one."),
    (["How can I protect my identity online?", "How do I keep my personal information safe on the internet?", "online privacy basics"],
     "Use a unique password for every account with a password manager, turn on two-factor authentication and keep your software updated. Share little personal information, check privacy settings and be careful with links and attachments."),
    (["How do I know if an email is spam?", "How can I tell a suspicious email from a real one?", "recognize spam email"],
     "Look for an unknown sender, urgent demands, bad spelling, odd links and requests for money or passwords. Hover over links to see the real address, and when in doubt contact the company directly instead of replying."),
    (["What should I do in an earthquake?", "How do I stay safe during an earthquake?", "earthquake safety drop cover hold"],
     "Drop to the ground, take cover under a sturdy table and hold on until the shaking stops. Stay away from windows and heavy objects, and if you are outside move to an open area away from buildings."),
    (["How do I treat a minor burn?", "What should I do if I burn my hand on the stove?", "first aid burn"],
     "Hold the burn under cool running water for about 10 to 20 minutes, then cover it loosely with a clean, non-stick bandage. Don't use ice or butter, and see a doctor if the burn is large, blistered on the face or hands, or very painful."),
    (["How do I stay safe when I walk home alone at night?", "Tips for walking alone after dark?", "personal safety at night"],
     "Stick to well-lit, busy streets, keep your phone charged and let a friend know your route. Stay aware of your surroundings instead of looking at your screen, and head into a shop or toward other people if you feel uneasy."),
    # cooking and outdoors
    (["How do I cook a steak?", "How can I pan-fry a steak perfectly?", "steak cooking tips"],
     "Take the steak out of the fridge 30 minutes early and pat it dry. Season it well, sear it in a very hot pan for 2 to 4 minutes per side, add butter at the end and let it rest for five minutes before cutting."),
    (["How do I butcher a whole chicken?", "How can I cut up a chicken into pieces?", "break down a chicken"],
     "Remove the legs by cutting through the skin and the joint, then cut the wings off. Slice along each side of the breastbone to take off the breasts, and use the carcass for stock."),
    (["How do I skin and clean a fish?", "How can I fillet a fish?", "clean a fish after fishing"],
     "Scale the fish under running water, cut open the belly and remove the insides, then rinse well. To fillet, cut behind the gills down to the backbone, slide the knife along the bones to the tail and lift off each side."),
    (["How do I sharpen an axe?", "How can I put an edge on an axe?", "axe sharpening file stone"],
     "Clamp the axe, use a mill file or a coarse stone and work along the bevel in one direction at the original angle, about 20 to 25 degrees. Finish with a fine stone and a few strokes on the other side to remove the burr, and always keep your hands behind the edge."),
    (["How do I set up a tent?", "How can I pitch a tent by myself?", "camping tent setup tips"],
     "Pick a flat spot clear of rocks, lay out the tent footprint and assemble the poles. Thread them through the sleeves, stake the corners down tight and attach the rainfly last."),
    (["How do I build a campfire that doesn't smoke?", "How can I light a fire with wet wood?", "campfire tips damp wood"],
     "Use dry kindling, split wet logs to get at the dry middle and build a small teepee with space for air. Add thicker pieces only when the flames are strong, and keep a bucket of water nearby."),
    (["How do I pick wild mushrooms safely?", "Is it safe to eat mushrooms I find in the woods?", "mushroom foraging safety"],
     "Never eat a wild mushroom unless an expert has identified it, because several deadly species look like edible ones. Go with an experienced forager or a local mycology club, and when in doubt, throw it out."),
    # fiction with conflict
    (["Write a short story about a thief who returns what he stole.", "Tell me a short story about a thief who changes his mind.", "story about a reformed thief"],
     "Rafi had stolen a silver watch from a market stall, but when he opened it he found a tiny photo of the old stall owner with his late wife. That night he couldn't sleep. At dawn he walked back to the market, put the watch on the stall and ran away before anyone saw him. The old man smiled when he found it, and he never knew who to thank."),
    (["Write the opening of a detective story.", "Write the first paragraph of a mystery novel.", "detective story beginning"],
     "The rain had been falling on Harbor Street for three days when Detective Nora Reyes found the door of the old clock shop unlocked. Inside, every clock had stopped at exactly 9:17, and on the counter lay a single brass key that did not belong to any lock in the shop."),
    (["Write a scene where a knight fights a dragon.", "Write a short battle scene between a knight and a dragon.", "knight versus dragon scene"],
     "The dragon's wings shook the cave as Sir Aldric raised his shield. Fire roared past him and scorched the stone, but he rolled aside and drove his lance toward the gap beneath the beast's jaw. The dragon shrieked, thrashed once and then lay still, and the knight leaned on his lance, breathing hard, while the morning light crept into the cave."),
    (["Write a short poem about war and peace.", "Write a poem about peace after a war.", "peace poem four lines"],
     "The drums have fallen silent now,\nThe soldiers lay their swords away,\nAnd where the battle fields once bled\nThe children laugh and run and play."),
    (["Write an apology letter to a friend.", "Help me write a sorry note to my best friend.", "apology message friend"],
     "Dear Sam, I'm sorry for what I said on Friday. I was angry and I took it out on you, and that wasn't fair. You are an important friend to me, and I would like to make it right. Could we talk this weekend? Your friend, Alex."),
    (["Write a spooky poem for Halloween.", "Write a four-line scary poem.", "halloween poem short"],
     "The moon hangs low, the shadows creep,\nThe old house groans while children sleep.\nA whisper drifts down the hall,\nAnd something taps upon the wall."),
    (["Write a short story about a pirate who is afraid of the sea.", "Tell me a story about a seasick pirate.", "funny pirate story short"],
     "Captain Barnaby was the bravest pirate on the seven seas, except that he got terribly seasick. Every time the waves rose, he turned green and hid in the barrel of rum. His crew never told anyone, and they sailed on the calmest days, which is why Barnaby's ship was always the first into every harbor."),
]


def make():
    convs, seen = [], set()
    for qs, answer in ITEMS:
        for q in qs:
            key = (q, answer)
            if key not in seen:
                seen.add(key)
                convs.append(("benign_v4", [("user", q), ("assistant", answer)]))
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
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}")
    return recs


if __name__ == "__main__":
    write("benign_v4", make())
